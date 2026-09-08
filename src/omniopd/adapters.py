from __future__ import annotations

import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone
from typing import Any, Literal, Sequence

from .protocol import ResetResult, StepResult
from .provenance import sha256_json


def validate_provider_response_model_identity(
    request_ledger: Sequence[dict[str, Any]],
    requested_model: str,
    *,
    require_success: bool = True,
) -> list[str]:
    """Require every successful response to attest the requested model alias."""

    model = str(requested_model).strip()
    if not model:
        raise ValueError("requested_model must be non-empty")
    successful = [row for row in request_ledger if row.get("status") == "ok"]
    if require_success and not successful:
        raise RuntimeError(
            "no successful provider response is available to attest model identity"
        )
    mismatches = [
        {
            "request_id": row.get("request_id"),
            "response_model": row.get("response_model"),
        }
        for row in successful
        if row.get("response_model") != model
    ]
    if mismatches:
        raise RuntimeError(
            "provider response.model does not equal the requested stable model alias: "
            f"requested={model!r}, mismatches={mismatches[:5]}"
        )
    return sorted({str(row["response_model"]) for row in successful})


class OpenAIChatPolicy:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        extra_body: dict[str, Any] | None = None,
        thinking_mode: Literal["enabled", "disabled"] | None = None,
        thinking_control: Literal["deepseek", "chat_template"] = "deepseek",
        reasoning_effort: Literal["low", "high", "max"] | None = None,
        max_retries: int = 0,
        seed: int | None = None,
        request_ledger_path: str | Path | None = None,
    ) -> None:
        from openai import OpenAI

        if thinking_control not in {"deepseek", "chat_template"}:
            raise ValueError(f"unsupported thinking_control={thinking_control!r}")
        if max_retries != 0:
            raise ValueError("exact request accounting requires max_retries=0")
        if seed is not None and seed < 0:
            raise ValueError("seed must be non-negative")
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.model = model
        self.base_url = base_url
        self.thinking_control = thinking_control
        if thinking_mode is None and reasoning_effort is not None:
            raise ValueError("reasoning_effort requires an explicit thinking_mode")
        if thinking_mode == "disabled" and reasoning_effort is not None:
            raise ValueError("reasoning_effort is invalid when thinking is disabled")
        if thinking_control == "chat_template" and reasoning_effort is not None:
            raise ValueError("reasoning_effort is only supported by DeepSeek thinking control")
        controlled_key = "thinking" if thinking_control == "deepseek" else "chat_template_kwargs"
        if extra_body and controlled_key in extra_body and thinking_mode is not None:
            raise ValueError(f"set thinking_mode instead of duplicating extra_body.{controlled_key}")
        self.extra_body = dict(extra_body or {})
        if thinking_mode is not None:
            if thinking_control == "deepseek":
                self.extra_body["thinking"] = {"type": thinking_mode}
            else:
                self.extra_body["chat_template_kwargs"] = {
                    "enable_thinking": thinking_mode == "enabled"
                }
        self.thinking_mode = thinking_mode
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.request_ledger: list[dict[str, Any]] = []
        self._request_ids: set[str] = set()
        self.request_ledger_path = (
            Path(request_ledger_path) if request_ledger_path is not None else None
        )
        if self.request_ledger_path is not None:
            if self.request_ledger_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite request ledger: {self.request_ledger_path}"
                )
            self.request_ledger_path.parent.mkdir(parents=True, exist_ok=True)
            self.request_ledger_path.touch(exist_ok=False)

    def _record_request(self, entry: dict[str, Any]) -> None:
        entry = {
            **entry,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.request_ledger.append(entry)
        if self.request_ledger_path is None:
            return
        with self.request_ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None,
        max_tokens: int,
        request_id: str,
    ) -> str:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if temperature is not None and not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be None or in [0,2]")
        if not request_id or any(character in request_id for character in "\r\n"):
            raise ValueError("request_id must be a non-empty single-line identifier")
        if request_id in self._request_ids:
            raise ValueError(f"duplicate request_id would corrupt call accounting: {request_id}")
        if self.thinking_mode == "enabled" and temperature is not None:
            if self.thinking_control == "deepseek":
                raise ValueError(
                    "temperature must be omitted in DeepSeek thinking mode because it is ignored"
                )
        self._request_ids.add(request_id)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "extra_headers": {"X-Request-ID": request_id},
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        message_hash = sha256_json(list(messages))
        ledger_entry: dict[str, Any] = {
            "request_id": request_id,
            "requested_model": self.model,
            "base_url": self.base_url,
            "messages_sha256": message_hash,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "thinking_mode": self.thinking_mode or "provider_default",
            "thinking_control": self.thinking_control,
            "reasoning_effort": self.reasoning_effort,
            "seed": self.seed,
            "sdk_max_retries": 0,
        }
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as error:
            ledger_entry.update(
                {"status": "error", "error_type": type(error).__name__}
            )
            self._record_request(ledger_entry)
            raise
        first_choice = response.choices[0] if response.choices else None
        usage = getattr(response, "usage", None)
        ledger_entry.update(
            {
                "status": "ok",
                "provider_request_id": getattr(response, "id", None),
                "response_model": getattr(response, "model", None),
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "finish_reason": getattr(first_choice, "finish_reason", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        )
        self._record_request(ledger_entry)
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""


def extract_task_description(initial_observation: str) -> str:
    """Extract Task while keeping the complete initial observation separately."""

    text = str(initial_observation)
    match = re.search(
        r"(?:your\s+task\s+is\s+to\s*:?\s*|your\s+task\s+is\s*:?\s*|task\s*:\s*)"
        r"(.+?)(?:\.|\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match and match.group(1).strip():
        return match.group(1).strip()
    parts = [part.strip() for part in text.split("\n\n") if part.strip()]
    return parts[-1] if len(parts) > 1 else text.strip()


def extract_task_type(game_file: str) -> str:
    """Extract the ALFWorld task family from its trial-directory name.

    Canonical ALFWorld paths end in
    ``<task-family>-.../trial_<id>/game.tw-pddl``. Prefer the authoritative
    sibling ``traj_data.json`` and use the task-family directory as a fallback.
    """

    path = Path(str(game_file))
    trajectory_metadata = path.with_name("traj_data.json")
    if trajectory_metadata.is_file():
        try:
            task_type = str(
                json.loads(trajectory_metadata.read_text(encoding="utf-8"))["task_type"]
            ).strip()
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ValueError(
                f"invalid ALFWorld trajectory metadata: {trajectory_metadata}"
            ) from error
        if task_type:
            return task_type

    parent = path.parent
    task_directory = parent.parent.name if parent.name.startswith("trial_") else parent.name
    if not task_directory or task_directory in {".", ".."}:
        raise ValueError(f"cannot infer ALFWorld task type from {game_file!r}")
    task_type = task_directory.split("-", 1)[0].strip()
    if not task_type:
        raise ValueError(f"cannot infer ALFWorld task type from {game_file!r}")
    return task_type


def unwrap_batched_info(info: dict[str, Any], index: int = 0) -> dict[str, Any]:
    output = {}
    for key, value in info.items():
        if isinstance(value, (str, bytes, dict)):
            output[key] = value
            continue
        try:
            output[key] = value[index]
        except (IndexError, KeyError, TypeError):
            output[key] = value
    return output


class AlfworldEnvironment:
    """One-game adapter. Imports ALFWorld/TextWorld only when instantiated."""

    def __init__(
        self,
        env_config: dict[str, Any],
        game_file: str,
        *,
        rollout_seed: int | None = None,
    ) -> None:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        request_infos = textworld.EnvInfos(
            won=True, admissible_commands=True, extras=["gamefile"]
        )
        method = env_config["general"]["training_method"]
        if method == "dagger":
            max_steps = env_config["dagger"]["training"]["max_nb_steps_per_episode"]
        elif method == "dqn":
            max_steps = env_config["rl"]["training"]["max_nb_steps_per_episode"]
        else:
            raise ValueError(f"unsupported training_method={method!r}")
        env_id = textworld.gym.register_games(
            [game_file],
            request_infos,
            batch_size=1,
            auto_reset=False,
            asynchronous=False,
            max_episode_steps=max_steps,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        )
        self._env = textworld.gym.make(env_id)
        self.game_file = str(game_file)
        self.rollout_seed = rollout_seed
        self.seed_attestation: dict[str, Any] | None = None
        if rollout_seed is not None:
            if rollout_seed < 0 or rollout_seed >= 2**32:
                self._env.close()
                raise ValueError("rollout_seed must be in [0, 2**32)")
            seed_method = getattr(self._env, "seed", None)
            if not callable(seed_method):
                self._env.close()
                raise RuntimeError(
                    "the installed TextWorld Gym environment cannot accept an explicit "
                    "rollout seed; refusing a confirmatory rollout"
                )
            try:
                seed_method(rollout_seed)
            except Exception as error:
                self._env.close()
                raise RuntimeError(
                    "TextWorld rejected the explicit environment rollout seed"
                ) from error
            self.seed_attestation = {
                "method": "textworld_gym_env.seed",
                "seed": rollout_seed,
                "seeded_before_first_reset": True,
                "alfred_demangler_shuffle": False,
            }

    def reset(self) -> ResetResult:
        observations, info = self._env.reset()
        unpacked = unwrap_batched_info(info)
        initial = str(observations[0])
        game_id = str(unpacked.get("extra.gamefile") or self.game_file)
        return ResetResult(
            observation=initial,
            admissible_actions=tuple(unpacked.get("admissible_commands") or ()),
            game_id=game_id,
            task_type=extract_task_type(game_id),
            task=extract_task_description(initial),
        )

    def step(self, action: str) -> StepResult:
        observations, _, dones, infos = self._env.step([action])
        unpacked = unwrap_batched_info(infos)
        return StepResult(
            observation=str(observations[0]),
            admissible_actions=tuple(unpacked.get("admissible_commands") or ()),
            done=bool(dones[0]),
            won=bool(unpacked.get("won", False)),
        )

    def close(self) -> None:
        self._env.close()


def list_alfworld_games(env_config: dict[str, Any], split: str) -> list[str]:
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    if split not in {"train", "eval_in_distribution", "eval_out_of_distribution"}:
        raise ValueError(f"unsupported split: {split}")
    environment = AlfredTWEnv(env_config, train_eval=split)
    games = list(environment.game_files)
    if not games:
        raise RuntimeError(f"ALFWorld returned no games for split {split!r}")
    return games
