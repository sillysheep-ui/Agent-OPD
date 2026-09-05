#!/usr/bin/env python3
"""Collect Route-A on-policy distillation data from ALFWorld.
 
Design:
  1) Student (local OpenAI-compatible endpoint, e.g. vLLM) rolls out on ALFWorld TRAIN games.
  2) Each turn stores the exact Student-state chat context before the Student response.
  3) Select M turns per episode (random by default; mean-NLL uncertainty is optional).
  4) Query a black-box Teacher N times on the same Student-state context.
  5) Keep only Teacher samples whose parsed action exactly matches the current
     ALFWorld admissible-command set; write selected-turn JSONL records.
 
This script intentionally does NOT perform training. It isolates data collection from
parameter updates so each OPD round is auditable and reproducible.
"""
from __future__ import annotations
 
import argparse
import copy
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
 
import yaml
from openai import OpenAI
 
 
SYSTEM_PROMPT = """You are an ALFWorld household agent. Solve the current task one environment turn at a time.
Reply in this format:
Thought: <brief reasoning>
Action: <one executable ALFWorld command>
Use the command grammar shown by the environment (for example go to, take, move, open, close, use, clean, heat, cool).
To place objects use "move X to Y" (NOT "put X in/on Y"). Object and receptacle names include their numbers, e.g. "fridge 1", "drawer 2".
Do not simulate future observations or future turns."""
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Route-A black-box OPD samples on ALFWorld train games")
    p.add_argument("--env-config", required=True, help="ALFWorld YAML config used by your rebuilt environment")
    p.add_argument("--output", required=True, help="Selected-turn JSONL output")
    p.add_argument("--limit-games", type=int, default=20, help="Number of train games to collect")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--turns-per-episode", type=int, default=3)
    p.add_argument("--selection", choices=["random", "mean_nll"], default="random",
                   help="First experiment should normally use random; mean_nll requires Student logprobs")
 
    p.add_argument("--student-base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--student-api-key", default="EMPTY")
    p.add_argument("--student-model", required=True,
                   help="Model name exposed by local vLLM/OpenAI-compatible server")
    p.add_argument("--student-temperature", type=float, default=0.0)
    p.add_argument("--student-max-tokens", type=int, default=256)
 
    p.add_argument("--teacher-base-url", default="https://api.deepseek.com")
    p.add_argument("--teacher-api-key", default=None,
                   help="Prefer env DEEPSEEK_API_KEY or --teacher-key-file instead of shell history")
    p.add_argument("--teacher-key-file", default="/root/.deepseek_key")
    p.add_argument("--teacher-model", default="deepseek-v4-flash")
    p.add_argument("--teacher-temperature", type=float, default=0.5,
                   help="Fix this across N ablations; it is part of the Teacher policy definition")
    p.add_argument("--teacher-max-tokens", type=int, default=3000,
                   help="DeepSeek v4 is a reasoning model: small max_tokens leaves content empty on long histories")
    p.add_argument("--teacher-samples", type=int, default=1, help="N teacher samples per selected turn")
 
    p.add_argument("--target-mode", choices=["full", "action"], default="full",
                   help="full keeps Teacher step text but canonicalizes Action; action trains only 'Action: <cmd>'")
    p.add_argument("--action-marker", choices=["first", "last"], default="first",
                   help="Use first to avoid Qwen whole-trajectory simulation contaminating the executed action")
    return p.parse_args()
 
 
def read_secret(cli_value: str | None, key_file: str | None, env_name: str) -> str:
    if cli_value:
        return cli_value.strip()
    env_value = os.getenv(env_name)
    if env_value:
        return env_value.strip()
    if key_file and Path(key_file).exists():
        raw = Path(key_file).read_text(encoding="utf-8").strip()
        # Allow "VAR=sk-..." or bare "sk-..." file formats.
        if "=" in raw and not raw.startswith("sk-"):
            raw = raw.split("=", 1)[1]
        return raw.strip()
    raise RuntimeError(f"Missing API key: pass CLI value, set {env_name}, or provide a readable key file")
 
 
def task_description(initial_observation: str) -> str:
    # Matches RoMeRL's released ALFWorld evaluator behavior: discard the welcome paragraph.
    parts = str(initial_observation).split("\n\n")
    return "\n".join(parts[1:]).strip() if len(parts) > 1 else str(initial_observation).strip()
 
 
def normalize_action(text: str) -> str:
    text = str(text or "").strip()
    text = text.strip("`\"' ")
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".;")
    # ALFWorld grammar dialect: "put X in/on/into Y" -> "move X to Y" (deterministic).
    m = re.match(r"^put\s+(.+?)\s+(?:in|on|into|onto)\s+(.+)$", text, re.IGNORECASE)
    if m:
        text = f"move {m.group(1)} to {m.group(2)}"
    return text.strip().lower()
 
 
def extract_action(response: str, marker_strategy: str = "first") -> str:
    """Extract one action line without fuzzy correction.
 
    For safety against models that simulate an entire trajectory, `first` takes the first
    Action: marker. If no marker exists, use the first non-empty line.
    """
    text = str(response or "").strip()
    matches = list(re.finditer(r"(?im)^\s*Action\s*:\s*", text))
    if matches:
        m = matches[0] if marker_strategy == "first" else matches[-1]
        tail = text[m.end():]
        return tail.splitlines()[0].strip()
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""
 
 
def canonical_action(candidate: str, admissible: Iterable[str]) -> str | None:
    # Exact normalized matching first; then unique-prefix completion (e.g. "go to sinkbasin"
    # -> "go to sinkbasin 1" when that is the ONLY admissible command extending it).
    # Both rules are deterministic; no fuzzy matching = no hidden task-specific judge.
    table = {normalize_action(a): str(a).strip() for a in admissible}
    norm = normalize_action(candidate)
    if norm in table:
        return table[norm]
    prefix_hits = [cmd for cmd in admissible if normalize_action(cmd).startswith(norm + " ")]
    if len(prefix_hits) == 1:
        return str(prefix_hits[0]).strip()
    return None
 
 
def canonicalize_teacher_response(raw: str, action: str, mode: str) -> str:
    if mode == "action":
        return f"Action: {action}"
    text = str(raw or "").strip()
    # Preserve Teacher's step text, but replace everything from the first Action marker onward
    # with the environment-canonical command. This keeps reasoning while making the action exact.
    m = re.search(r"(?im)^\s*Action\s*:\s*", text)
    if m:
        prefix = text[:m.start()].rstrip()
        if prefix:
            return f"{prefix}\nAction: {action}"
    return f"Action: {action}"
 
 
def completion_mean_nll(choice: Any) -> float | None:
    """Approximate Student uncertainty using generated-token mean NLL.
 
    This is NOT full action entropy. It is only an optional Student-only uncertainty proxy.
    """
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    vals: list[float] = []
    if content:
        for tok in content:
            v = getattr(tok, "logprob", None)
            if v is not None and math.isfinite(float(v)):
                vals.append(-float(v))
    return sum(vals) / len(vals) if vals else None
 
 
def chat_generate(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    want_logprobs: bool = False,
) -> tuple[str, float | None]:
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if want_logprobs:
        kwargs.update(logprobs=True, top_logprobs=5)
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    content = choice.message.content or ""
    return content, completion_mean_nll(choice) if want_logprobs else None
 
 
@dataclass
class TurnSnapshot:
    turn_index: int
    query_messages: list[dict[str, str]]
    observation: str
    admissible_commands: list[str]
    student_response: str
    student_action: str
    student_action_valid: bool
    uncertainty_mean_nll: float | None
 
 
def select_turns(turns: list[TurnSnapshot], m: int, mode: str, rng: random.Random) -> list[TurnSnapshot]:
    if not turns or m <= 0:
        return []
    m = min(m, len(turns))
    if mode == "random":
        return sorted(rng.sample(turns, k=m), key=lambda x: x.turn_index)
    missing = [t.turn_index for t in turns if t.uncertainty_mean_nll is None]
    if missing:
        raise RuntimeError(f"mean_nll selection requested but logprobs missing for turns: {missing}")
    return sorted(sorted(turns, key=lambda x: float(x.uncertainty_mean_nll), reverse=True)[:m],
                  key=lambda x: x.turn_index)
 
 
def list_train_games(env_config: dict[str, Any]) -> list[str]:
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    controller = AlfredTWEnv(env_config, train_eval="train")
    return list(controller.game_files)
 
 
def build_one_game_env(env_config: dict[str, Any], game: str):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
 
    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        extras=["gamefile"],
    )
    method = env_config["general"]["training_method"]
    if method == "dagger":
        max_episode_steps = env_config["dagger"]["training"]["max_nb_steps_per_episode"]
    elif method == "dqn":
        max_episode_steps = env_config["rl"]["training"]["max_nb_steps_per_episode"]
    else:
        raise ValueError(f"Unsupported ALFWorld training_method={method!r}; use dqn for collection")
 
    env_id = textworld.gym.register_games(
        [game],
        request_infos,
        batch_size=1,
        auto_reset=False,
        asynchronous=False,
        max_episode_steps=max_episode_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
    )
    return textworld.gym.make(env_id)
 
 
def unwrap_info(info: dict[str, Any], index: int = 0) -> dict[str, Any]:
    out = {}
    for key, value in info.items():
        try:
            out[key] = value[index]
        except Exception:
            out[key] = value
    return out
 
 
def collect_episode(
    *,
    game: str,
    env_config: dict[str, Any],
    student: OpenAI,
    student_model: str,
    student_temperature: float,
    student_max_tokens: int,
    max_steps: int,
    selection: str,
    action_marker: str,
) -> tuple[list[TurnSnapshot], dict[str, Any]]:
    env = build_one_game_env(env_config, game)
    try:
        observations, info = env.reset()
        obs = str(observations[0])
        info0 = unwrap_info(info)
        desc = task_description(obs)
        history: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Now solve this new task.\n{desc}"},
        ]
        current_observation = desc
        turns: list[TurnSnapshot] = []
        success = False
        done = False
 
        for step in range(max_steps):
            admissible = list(info0.get("admissible_commands") or [])
            query_messages = copy.deepcopy(history)
            if step > 0:
                query_messages.append({"role": "user", "content": f"Observation: {current_observation.strip()}"})
 
            response, nll = chat_generate(
                student,
                model=student_model,
                messages=query_messages,
                temperature=student_temperature,
                max_tokens=student_max_tokens,
                want_logprobs=(selection == "mean_nll"),
            )
            proposed = extract_action(response, action_marker)
            canonical = canonical_action(proposed, admissible)
            # Route A design: a_t^S = g(Y_t^S) maps free text to ONE legal command.
            # Never execute raw invalid text; fall back to a safe no-op action.
            execute_action = canonical if canonical is not None else "look"
 
            turns.append(TurnSnapshot(
                turn_index=step,
                query_messages=query_messages,
                observation=current_observation,
                admissible_commands=admissible,
                student_response=response,
                student_action=execute_action,
                student_action_valid=(canonical is not None),
                uncertainty_mean_nll=nll,
            ))
 
            # Update Student history using ONLY the response that was actually generated at this turn.
            if step > 0:
                history.append({"role": "user", "content": f"Observation: {current_observation.strip()}"})
            history.append({"role": "assistant", "content": response})
 
            observations, rewards, dones, infos = env.step([execute_action])
            current_observation = str(observations[0])
            info0 = unwrap_info(infos)
            done = bool(dones[0])
            won = bool(info0.get("won", False))
            reward = float(rewards[0]) if rewards is not None else 0.0
            success = won or reward > 0
            if done:
                break
 
        meta = {
            "gamefile": str(info0.get("extra.gamefile") or game),
            "task_description": desc,
            "student_success": bool(success),
            "student_steps": len(turns),
            "student_invalid_actions": sum(not t.student_action_valid for t in turns),
        }
        return turns, meta
    finally:
        try:
            env.close()
        except Exception:
            pass
 
 
def teacher_samples_for_turn(
    *,
    teacher: OpenAI,
    model: str,
    turn: TurnSnapshot,
    temperature: float,
    max_tokens: int,
    n: int,
    action_marker: str,
    target_mode: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for i in range(n):
        raw, _ = chat_generate(
            teacher,
            model=model,
            messages=turn.query_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            want_logprobs=False,
        )
        proposed = extract_action(raw, action_marker)
        canonical = canonical_action(proposed, turn.admissible_commands)
        valid = canonical is not None
        samples.append({
            "sample_index": i,
            "raw_response": raw,
            "parsed_action": proposed,
            "canonical_action": canonical,
            "valid": valid,
            "target_response": canonicalize_teacher_response(raw, canonical, target_mode) if valid else None,
        })
    return samples
 
 
def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
 
    env_cfg = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    method = env_cfg.get("general", {}).get("training_method")
    if method != "dqn":
        print(f"WARNING: env general.training_method={method!r}; your handoff recommends dqn during collection/eval.")
 
    student = OpenAI(api_key=args.student_api_key, base_url=args.student_base_url)
    teacher_key = read_secret(args.teacher_api_key, args.teacher_key_file, "DEEPSEEK_API_KEY")
    teacher = OpenAI(api_key=teacher_key, base_url=args.teacher_base_url)
 
    games = list_train_games(env_cfg)
    rng.shuffle(games)
    if args.limit_games is not None:
        games = games[: max(0, args.limit_games)]
 
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
 
    total_selected = total_valid_teacher = total_teacher = 0
    total_student_success = 0
 
    with out_path.open("w", encoding="utf-8") as fout:
        for ep_idx, game in enumerate(games):
            turns, meta = collect_episode(
                game=game,
                env_config=env_cfg,
                student=student,
                student_model=args.student_model,
                student_temperature=args.student_temperature,
                student_max_tokens=args.student_max_tokens,
                max_steps=args.max_steps,
                selection=args.selection,
                action_marker=args.action_marker,
            )
            total_student_success += int(meta["student_success"])
            selected = select_turns(turns, args.turns_per_episode, args.selection, rng)
 
            for turn in selected:
                samples = teacher_samples_for_turn(
                    teacher=teacher,
                    model=args.teacher_model,
                    turn=turn,
                    temperature=args.teacher_temperature,
                    max_tokens=args.teacher_max_tokens,
                    n=args.teacher_samples,
                    action_marker=args.action_marker,
                    target_mode=args.target_mode,
                )
                total_selected += 1
                total_teacher += len(samples)
                total_valid_teacher += sum(bool(x["valid"]) for x in samples)
 
                record = {
                    "episode_index": ep_idx,
                    **meta,
                    "turn": asdict(turn),
                    "selection": args.selection,
                    "teacher_model": args.teacher_model,
                    "teacher_temperature": args.teacher_temperature,
                    "teacher_samples_n": args.teacher_samples,
                    "target_mode": args.target_mode,
                    "teacher_samples": samples,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
 
            print(json.dumps({
                "episode": ep_idx + 1,
                "games": len(games),
                "success": meta["student_success"],
                "steps": meta["student_steps"],
                "invalid_student_actions": meta["student_invalid_actions"],
                "selected_turns": [t.turn_index for t in selected],
            }, ensure_ascii=False))
 
    print(json.dumps({
        "output": str(out_path),
        "episodes": len(games),
        "student_successes": total_student_success,
        "student_success_rate": total_student_success / len(games) if games else 0.0,
        "selected_turns": total_selected,
        "teacher_calls": total_teacher,
        "valid_teacher_samples": total_valid_teacher,
        "teacher_valid_rate": total_valid_teacher / total_teacher if total_teacher else 0.0,
    }, ensure_ascii=False, indent=2))
 
 
if __name__ == "__main__":
    main()
