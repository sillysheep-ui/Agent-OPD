#!/usr/bin/env python3
"""Requirement-conformant MVP of the black-box Agent OPD data path.

Implements AGENT_OPD.md steps 1-4 with the conventions the registered chain
uses, and prints a conformance line per requirement so a mismatch is visible:

  1. states    : rebuild AgentState objects from the collected Student pool
  2. selection : `uniform_per_game_nested_v1` semantics -- per game, states are
                 ranked by a seeded hash priority and the first m are taken
                 (nested across m), with inclusion probability m/T_g recorded
  3. teacher   : text-only query to an OpenAI-compatible endpoint with the
                 Teacher system prompt (P_T != P_S), temperature from the
                 registered profile, N attempts per state, attempt ledger plus
                 `messages_sha256` verification against the request ledger
  4. rows      : CorrectionRecord -> build_training_rows(game_state_mean)
                 -> action-only token masks

Acceptance is reported at sample, state and game level, which the objective's
`P(K_s>0|s)=1-(1-p_v)^N` retention argument requires.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import OpenAIChatPolicy  # noqa: E402
from omniopd.dataset import build_training_rows  # noqa: E402
from omniopd.parser import parse_action  # noqa: E402
from omniopd.prompts import TEACHER_SYSTEM_PROMPT, replace_system  # noqa: E402
from omniopd.provenance import sha256_file, sha256_json  # noqa: E402
from omniopd.schema import ActionSample, AgentState, CorrectionRecord  # noqa: E402
from omniopd.tokenization import encode_final_assistant_example  # noqa: E402


def load_episodes(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def student_sample(turn: dict) -> ActionSample:
    student = turn["student"]
    return ActionSample(
        raw=str(student.get("raw") or ""),
        executed_action=str(student.get("executed_action") or ""),
        valid=bool(student.get("valid")),
        had_action_marker=bool(student.get("had_action_marker")),
        failure_reason=student.get("failure_reason"),
    )


def agent_state(turn: dict) -> AgentState:
    state = turn["state"]
    return AgentState(
        task=state["task"],
        observation=state["observation"],
        admissible_actions=tuple(state["admissible_actions"]),
        messages=tuple(dict(message) for message in state["messages"]),
        game_id=state["game_id"],
        task_type=state["task_type"],
        turn_index=int(state["turn_index"]),
        full_messages=tuple(dict(message) for message in state.get("full_messages", state["messages"])),
        truncation=dict(state.get("truncation", {})),
        state_source="student",
    )


def priority(seed: int, game_id: str, state_hash: str) -> str:
    """Per-game hash priority, the ordering `uniform_per_game_nested_v1` uses."""

    return hashlib.sha256(f"{seed}:{game_id}:{state_hash}".encode("utf-8")).hexdigest()


def select_nested(episodes: list[dict], games: int, per_game: int, seed: int):
    """SRSWOR m-of-T per game with a shared priority order (nested across m)."""

    usable = [
        episode
        for episode in episodes
        if any(turn["state"].get("admissible_actions") for turn in episode.get("turns", []))
    ]
    if len(usable) < games:
        raise SystemExit(f"pool has {len(usable)} usable games, need {games}")
    order = sorted(usable, key=lambda episode: priority(seed, "game", episode["game_id"]))
    selected = []
    for episode in order[:games]:
        turns = [turn for turn in episode["turns"] if turn["state"].get("admissible_actions")]
        ranked = sorted(turns, key=lambda turn: priority(seed, episode["game_id"], turn["state"]["state_hash"]))
        for turn in ranked[:per_game]:
            selected.append((episode, turn, per_game, len(turns)))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--states-per-game", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-url", required=True)
    parser.add_argument("--teacher-model", default="teacher")
    parser.add_argument("--teacher-profile", type=Path,
                        default=ROOT / "configs" / "teacher_sampling_nonthinking.yaml")
    parser.add_argument("--student-tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    profile = yaml.safe_load(args.teacher_profile.read_text(encoding="utf-8"))
    temperature = profile.get("temperature")
    max_tokens = int(profile.get("max_tokens", 256))
    thinking_mode = str(profile.get("thinking_mode"))

    # ---- step 1 ---------------------------------------------------------
    episodes = load_episodes(args.pool)
    turns_total = sum(len(episode.get("turns", [])) for episode in episodes)
    print(f"[1] states      : {len(episodes)} episodes, {turns_total} turns")

    # ---- step 2 ---------------------------------------------------------
    selected = select_nested(episodes, args.games, args.states_per_game, args.seed)
    games_hit = sorted({episode["game_id"] for episode, _, _, _ in selected})
    print(f"[2] selection   : {len(selected)} states / {len(games_hit)} games, "
          f"policy=uniform_per_game_nested_v1 seed={args.seed} m={args.states_per_game} "
          f"inclusion_prob={args.states_per_game}/T_g")

    # ---- step 3 ---------------------------------------------------------
    policy = OpenAIChatPolicy(
        model=args.teacher_model,
        base_url=args.teacher_url,
        api_key="EMPTY",
        thinking_mode="disabled" if thinking_mode == "disabled" else "enabled",
        thinking_control="chat_template",
        max_retries=0,
        request_ledger_path=args.output_dir / "requests.jsonl",
    )
    records: list[CorrectionRecord] = []
    ledger_rows: list[dict] = []
    expected_hashes: dict[str, str] = {}
    for index, (episode, turn, m, t_g) in enumerate(selected):
        state = agent_state(turn)
        # P_T: the Teacher is asked in its own role, not the Student's.
        teacher_messages = replace_system(list(state.messages), TEACHER_SYSTEM_PROMPT)
        samples: list[ActionSample] = []
        for attempt in range(args.attempts):
            request_id = f"mvp-{index:04d}-{attempt}"
            expected_hashes[request_id] = sha256_json(list(teacher_messages))
            try:
                raw = policy.generate(
                    teacher_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_id=request_id,
                )
            except Exception as error:  # noqa: BLE001 - count the attempt, keep going
                samples.append(ActionSample(raw="", executed_action="", valid=False,
                                            had_action_marker=False,
                                            failure_reason=f"request_error:{type(error).__name__}"))
                ledger_rows.append({"request_id": request_id, "status": "error",
                                    "error": type(error).__name__, "valid": False})
                continue
            parsed = parse_action(raw, state.admissible_actions)
            samples.append(ActionSample(raw=raw,
                                        executed_action=parsed.canonical_action or parsed.normalized_candidate or "",
                                        valid=parsed.valid, had_action_marker=parsed.had_action_marker,
                                        failure_reason=parsed.failure_reason))
            ledger_rows.append({"request_id": request_id, "game_id": state.game_id,
                                "state_hash": state.state_hash, "attempt": attempt,
                                "valid": parsed.valid, "failure_reason": parsed.failure_reason,
                                "temperature": temperature, "raw": raw[:400]})
        records.append(CorrectionRecord(
            state=state,
            student=student_sample(turn),
            teacher_samples=samples,
            selection_policy="uniform_per_game_nested_v1",
            inclusion_probability=m / t_g,
            teacher_calls=args.attempts,
        ))

    # ledger hash verification: the request ledger must describe the messages
    # we actually sent, per request id.
    request_rows = {str(row["request_id"]): row for row in policy.request_ledger}
    hash_mismatch = [
        request_id
        for request_id, digest in expected_hashes.items()
        if request_id in request_rows and request_rows[request_id].get("messages_sha256") != digest
    ]
    missing = [rid for rid in expected_hashes if rid not in request_rows]

    attempts = len(ledger_rows)
    valid_attempts = sum(1 for row in ledger_rows if row.get("valid"))
    valid_states = sum(1 for record in records if record.valid_teacher_samples)
    valid_games = len({record.state.game_id for record in records if record.valid_teacher_samples})
    print(f"[3] teacher     : P_T applied, profile={profile.get('profile')} temperature={temperature} "
          f"N={args.attempts}")
    print(f"    budget      : attempts={attempts} request_ledger={len(request_rows)} "
          f"hash_verified={'yes' if not hash_mismatch and not missing else 'NO'} "
          f"(mismatch={len(hash_mismatch)} missing={len(missing)})")
    print(f"    acceptance  : sample {valid_attempts}/{attempts}, "
          f"state {valid_states}/{len(records)}, game {valid_games}/{len(games_hit)}")
    (args.output_dir / "records.jsonl").write_text(
        "\n".join(json.dumps({
            "state": record.state.to_dict(),
            "student": record.student.to_dict(),
            "teacher_samples": [s.to_dict() for s in record.teacher_samples],
            "selection_policy": record.selection_policy,
            "inclusion_probability": record.inclusion_probability,
            "teacher_calls": record.teacher_calls,
        }, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")
    (args.output_dir / "attempt_ledger.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in ledger_rows) + "\n", encoding="utf-8")

    # ---- step 4 ---------------------------------------------------------
    if hash_mismatch or missing:
        raise SystemExit("request ledger does not match the sent messages")
    rows = build_training_rows(records, weighting="game_state_mean")
    if not rows:
        raise SystemExit("no valid Teacher action produced any training row")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.student_tokenizer), local_files_only=True)
    encoded_rows = []
    for row in rows:
        messages = [*row.messages, {"role": "assistant", "content": f"Action: {row.teacher_action}"}]
        example = encode_final_assistant_example(tokenizer, messages, max_length=args.max_length,
                                                 truncation="error", enable_thinking=row.enable_thinking)
        encoded_rows.append({
            "input_ids": example.input_ids,
            "target_token_mask": example.target_token_mask,
            "state_weight": row.state_weight,
            "game_id": row.game_id,
            "state_hash": row.state_hash,
            "teacher_action": row.teacher_action,
            "student_action": row.student_action,
        })
    weight_by_game = defaultdict(float)
    for row in encoded_rows:
        weight_by_game[row["game_id"]] += row["state_weight"]
    (args.output_dir / "sft_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in encoded_rows) + "\n", encoding="utf-8")
    print(f"[4] rows        : {len(encoded_rows)} rows, "
          f"target tokens {sum(sum(r['target_token_mask']) for r in encoded_rows)}, "
          f"length {min(len(r['input_ids']) for r in encoded_rows)}-"
          f"{max(len(r['input_ids']) for r in encoded_rows)}")
    print("    weight/game : " + ", ".join(f"{g.split('/')[-1][:20]}={w:.3f}"
                                           for g, w in sorted(weight_by_game.items())))
    print(f"    pool sha256 : {sha256_file(args.pool)[:16]}…  output: {args.output_dir}")


if __name__ == "__main__":
    main()
