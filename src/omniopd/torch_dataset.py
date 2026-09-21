from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Sequence

from omniopd.tokenization import encode_final_assistant_content
from omniopd.prompts import resolve_student_prompt


class FinalTurnActionDataset:
    """Canonical final-turn dataset for PyTorch/veRL-style trainers.

    The Student system prompt version is read from the data config, so a
    prompt ablation cannot silently train against a different context than the
    one its audit recorded.

    Historical assistant actions remain context and always receive mask 0.
    `state_weight` remains a scalar so the trainer can normalize over the full
    optimizer batch rather than each microbatch.
    """

    def __init__(
        self,
        files: str | Path | Sequence[str | Path] | None = None,
        tokenizer: Any = None,
        config: Any | None = None,
        *,
        parquet_files: str | Path | Sequence[str | Path] | None = None,
        max_length: int = 4096,
        truncation: str = "error",
    ) -> None:
        import pandas as pd

        if tokenizer is None:
            raise ValueError("tokenizer is required")
        source = parquet_files if parquet_files is not None else files
        if source is None:
            raise ValueError("files/parquet_files is required")
        prompt_name = None
        if config is not None:
            getter = config.get if hasattr(config, "get") else lambda key, default: getattr(config, key, default)
            max_length = int(getter("max_length", max_length))
            truncation = str(getter("truncation", truncation))
            prompt_name = getter("student_prompt", None)
        self.student_prompt_name, self.student_prompt = resolve_student_prompt(prompt_name)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.truncation = truncation
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.truncation != "error":
            raise ValueError(
                "canonical training requires truncation='error'; rebuild the rollout "
                "state with TaskPreservingTruncator instead of deleting its prefix"
            )
        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be defined")
        paths = [source] if isinstance(source, (str, Path)) else list(source)
        frames = []
        for path in paths:
            path = Path(path)
            if path.suffix == ".parquet":
                frames.append(pd.read_parquet(path))
            elif path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    frames.append(
                        pd.DataFrame(
                            json.loads(line)
                            for line in handle
                            if line.strip()
                        )
                    )
            else:
                raise ValueError("training inputs must end in .parquet or .jsonl")
        self.frame = pd.concat(frames, ignore_index=True)
        required = {
            "messages",
            "state_weight",
            "state_hash",
            "game_id",
            "turn_index",
            "weighting_mode",
            "protocol_version",
            "enable_thinking",
        }
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"dataset is missing columns: {sorted(missing)}")
        has_expert_targets = {
            "target_sample_index",
            "target_action",
            "target_source",
        }.issubset(self.frame.columns)
        has_teacher_targets = {
            "teacher_sample_index",
            "teacher_action",
        }.issubset(self.frame.columns)
        if has_expert_targets == has_teacher_targets:
            raise ValueError(
                "dataset must contain exactly one target schema: generic target fields "
                "or legacy Teacher fields"
            )
        sample_index_column = (
            "target_sample_index" if has_expert_targets else "teacher_sample_index"
        )
        action_column = "target_action" if has_expert_targets else "teacher_action"
        if self.frame.empty:
            raise ValueError("training dataset is empty")
        identities = list(
            zip(
                self.frame["state_hash"].astype(str),
                self.frame[sample_index_column].astype(int),
            )
        )
        if len(identities) != len(set(identities)):
            raise ValueError("training dataset contains duplicate state/sample identities")
        versions = set(self.frame["protocol_version"].astype(str))
        if versions != {"omniopd-v1"}:
            raise ValueError(f"training dataset has unsupported protocol versions: {versions}")
        weighting_modes = set(self.frame["weighting_mode"].astype(str))
        if weighting_modes != {"game_state_mean"}:
            raise ValueError(
                "confirmatory training requires one game_state_mean weighting contract"
            )
        if any(bool(value) for value in self.frame["enable_thinking"]):
            raise ValueError("Student action-only targets must disable thinking")
        weights = self.frame["state_weight"].astype(float)
        if not all(math.isfinite(value) and value > 0.0 for value in weights):
            raise ValueError("stored state_weight values must be finite and strictly positive")
        self.objective_weight_sum = float(weights.sum())
        self.objective_population_size = int(len(self.frame))
        self.game_ids = frozenset(self.frame["game_id"].astype(str))
        for game, group in self.frame.assign(_weight=weights).groupby("game_id"):
            if not math.isclose(float(group["_weight"].sum()), 1.0, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(
                    f"game_state_mean weights for game {game!r} do not sum to one"
                )
        self._encoded = []
        for row_index, row in self.frame.iterrows():
            messages = _nested_to_python(row["messages"])
            roles = [message.get("role") for message in messages]
            expected_roles = [
                "system" if index == 0 else "user" if index % 2 else "assistant"
                for index in range(len(messages))
            ]
            expected_target = f"Action: {row[action_column]}"
            if (
                len(messages) < 3
                or roles != expected_roles
                or messages[0].get("content") != self.student_prompt
                or messages[-1] != {"role": "assistant", "content": expected_target}
            ):
                raise ValueError(
                    f"training row {row_index} violates the Student-context/final-action contract"
                )
            self._encoded.append(
                encode_final_assistant_content(
                    self.tokenizer,
                    messages,
                    max_length=self.max_length,
                    truncation=self.truncation,
                    enable_thinking=False,
                )
            )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        multiplier = 1.0
        if isinstance(index, tuple):
            index, multiplier = index
        if multiplier not in {0.0, 1.0}:
            raise ValueError(f"sampler multiplier must be 0 or 1, got {multiplier}")
        row = self.frame.iloc[index]
        state_weight = float(row["state_weight"]) * float(multiplier)
        if not math.isfinite(state_weight) or state_weight < 0:
            raise ValueError(f"invalid state_weight at row {index}: {state_weight}")
        encoded = self._encoded[index]
        input_ids = torch.tensor(encoded.input_ids, dtype=torch.long)
        attention = torch.tensor(encoded.attention_mask, dtype=torch.long)
        target_mask = torch.tensor(encoded.target_token_mask, dtype=torch.long)
        padding = self.max_length - len(input_ids)
        if padding:
            pad_id = self.tokenizer.pad_token_id
            input_ids = torch.cat([input_ids, torch.full((padding,), pad_id, dtype=torch.long)])
            attention = torch.cat([attention, torch.zeros(padding, dtype=torch.long)])
            target_mask = torch.cat([target_mask, torch.zeros(padding, dtype=torch.long)])
        position_ids = torch.arange(self.max_length, dtype=torch.long) * attention
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "position_ids": position_ids,
            "target_token_mask": target_mask,
            "state_weight": torch.tensor(state_weight, dtype=torch.float32),
            "sampling_multiplier": torch.tensor(float(multiplier), dtype=torch.float32),
        }


def _nested_to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _nested_to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nested_to_python(item) for item in value]
    if hasattr(value, "tolist"):
        return _nested_to_python(value.tolist())
    return value
