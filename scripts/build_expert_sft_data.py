#!/usr/bin/env python3
"""Turn collected handcoded-expert episodes into audited action-only SFT rows.

Plan items 2 and 3 of the cold-start chain:

- only episodes the expert actually solved (``won``) become training states;
- each expert turn becomes one row whose context is the truncated history the
  Student would see and whose single target is the last ``Action: <command>``;
- games are split as whole units, so no game contributes to both train and
  validation, and the evaluation splits never appear here because the
  collection manifest is restricted to the train split;
- rows carry ``game_state_mean`` weights, so every game contributes one unit of
  objective mass no matter how many turns it took;
- the produced files are re-parsed with the real ``FinalTurnActionDataset``
  (the same class the trainer uses) before the audit is written.

Nothing here trains and nothing calls an API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.dataset import split_rows_by_game_stratified  # noqa: E402
from omniopd.io import read_jsonl, rollout_turn_from_dict  # noqa: E402
from omniopd.prompts import resolve_student_prompt  # noqa: E402
from omniopd.provenance import (  # noqa: E402
    fingerprint_code_tree,
    git_revision,
    sha256_file,
    sha256_text,
)

TARGET_SOURCE = "alfworld_handcoded_expert"


def build_rows(
    episodes: list[dict], *, student_prompt: str | None = None
) -> tuple[list[dict], dict]:
    """Build training rows from solved episodes and report what was dropped."""

    rows: list[dict] = []
    counts = {
        "episodes_read": len(episodes),
        "episodes_kept": 0,
        "episodes_dropped_unsolved": 0,
        "turns_dropped_unsolved": 0,
        "invalid_target_turns": 0,
    }
    for episode in episodes:
        turns = episode.get("turns") or []
        if not episode.get("won"):
            counts["episodes_dropped_unsolved"] += 1
            counts["turns_dropped_unsolved"] += len(turns)
            continue
        if not turns:
            raise ValueError(f"solved episode has no turns: {episode.get('game_id')}")
        counts["episodes_kept"] += 1
        weight = 1.0 / len(turns)
        for step_index, raw_turn in enumerate(turns):
            turn = rollout_turn_from_dict(raw_turn)
            if not turn.student.valid:
                counts["invalid_target_turns"] += 1
                raise ValueError(
                    "a solved expert episode contains an invalid action at "
                    f"{turn.state.game_id}/t{turn.state.turn_index}: "
                    f"{turn.student.failure_reason}"
                )
            messages = [dict(message) for message in turn.state.messages]
            # The scripted expert ignores the prompt, so a prompt ablation
            # re-roles the same executed actions instead of re-collecting.
            if not messages or messages[0].get("role") != "system":
                raise ValueError(
                    f"episode turn does not start with a system message: "
                    f"{turn.state.game_id}/t{turn.state.turn_index}"
                )
            messages[0] = {"role": "system", "content": student_prompt}
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Action: {turn.student.executed_action}",
                }
            )
            rows.append(
                {
                    "messages": messages,
                    "state_weight": weight,
                    "state_hash": turn.state.state_hash,
                    "game_id": str(episode["game_id"]),
                    "task_type": str(episode["task_type"]),
                    "turn_index": int(turn.state.turn_index),
                    "episode_step": step_index,
                    "target_sample_index": 0,
                    "target_action": turn.student.executed_action,
                    "target_source": TARGET_SOURCE,
                    "weighting_mode": "game_state_mean",
                    "protocol_version": "omniopd-v1",
                    "enable_thinking": False,
                    "state_source": turn.state.state_source,
                }
            )
    if not rows:
        raise ValueError("no solved episode produced a training row")
    return rows, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-val", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--student-prompt",
        default="v1",
        help="Student system prompt version used for the rebuilt rows",
    )
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


def main() -> None:
    args = parse_args()
    episodes_path = args.episodes.resolve()
    collection_manifest_path = args.collection_manifest.resolve()
    tokenizer_path = args.tokenizer.resolve()
    train_path = args.output_train.resolve()
    val_path = args.output_val.resolve()
    audit_path = args.audit_output.resolve()
    if (
        len({episodes_path, collection_manifest_path, train_path, val_path, audit_path}) != 5
        or train_path == val_path
        or any(path.exists() for path in (train_path, val_path, audit_path))
        or not 0.0 < args.val_fraction < 1.0
        or args.split_seed < 0
        or args.max_length <= 0
        or not episodes_path.is_file()
        or not collection_manifest_path.is_file()
        or not (tokenizer_path / "tokenizer_config.json").is_file()
    ):
        raise SystemExit("invalid expert SFT data build input or output")

    collection = json.loads(collection_manifest_path.read_text(encoding="utf-8"))
    if (
        collection.get("artifact") != "expert_trajectory_collection"
        or collection.get("supervision_source") != TARGET_SOURCE
        or collection.get("split") != "train"
    ):
        raise SystemExit("collection manifest is not a train-split expert collection")
    recorded_episodes = collection.get("episodes") or {}
    if recorded_episodes.get("sha256") != sha256_file(episodes_path):
        raise SystemExit("episode file does not match the collection manifest hash")

    episodes = list(read_jsonl(episodes_path))
    prompt_name, prompt_text = resolve_student_prompt(args.student_prompt)
    rows, counts = build_rows(episodes, student_prompt=prompt_text)
    # Stratify by task family so every family reaches the validation side;
    # a global split can leave whole families without any closed-loop game.
    train_rows, val_rows, split_metadata = split_rows_by_game_stratified(
        rows,
        val_fraction=args.val_fraction,
        rng_seed=args.split_seed,
        stratum_key=lambda row: row["task_type"],
        game_key=lambda row: row["game_id"],
    )
    if not train_rows or not val_rows:
        raise SystemExit("game split produced an empty train or validation side")
    write_rows(train_path, train_rows)
    write_rows(val_path, val_rows)

    from transformers import AutoTokenizer

    from omniopd.torch_dataset import FinalTurnActionDataset

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    checked = {}
    for name, path in (("train", train_path), ("validation", val_path)):
        dataset = FinalTurnActionDataset(
            files=path, tokenizer=tokenizer, max_length=args.max_length
        )
        checked[name] = {
            "rows": len(dataset),
            "games": len(dataset.game_ids),
            "objective_weight_sum": dataset.objective_weight_sum,
            "max_length": dataset.max_length,
        }

    audit = {
        "artifact": "expert_sft_data_audit",
        "protocol_version": "omniopd-v1",
        "training_ready": True,
        "confirmatory_use_allowed": False,
        "supervision_source": TARGET_SOURCE,
        "target_format": "Action: <executed command>",
        "weighting_mode": "game_state_mean",
        "student_prompt": {
            "name": prompt_name,
            "sha256": sha256_text(prompt_text),
        },
        "code_revision": git_revision(ROOT),
        "code": fingerprint_code_tree(ROOT),
        "script_sha256": sha256_file(Path(__file__)),
        "episodes": {
            "path": str(episodes_path),
            "sha256": sha256_file(episodes_path),
        },
        "collection_manifest": {
            "path": str(collection_manifest_path),
            "sha256": sha256_file(collection_manifest_path),
            "game_artifacts_digest": collection.get("game_artifacts_digest"),
            "environment_master_seed": collection.get("environment_master_seed"),
        },
        "split": {
            **split_metadata,
            "seed": args.split_seed,
            "val_fraction": args.val_fraction,
        },
        "counts": {
            **counts,
            "rows": len(rows),
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "train_games": len({row["game_id"] for row in train_rows}),
            "validation_games": len({row["game_id"] for row in val_rows}),
        },
        "dataset_self_check": checked,
        "outputs": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "validation": {"path": str(val_path), "sha256": sha256_file(val_path)},
        },
        "notes": [
            "split is stratified by task family: every family contributes whole games to validation",
            "evaluation games come from the valid_seen/valid_unseen splits and never appear here",
            "an unsolved expert episode contributes no training row; its turn count is recorded above",
        ],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        f"SFT rows: train={len(train_rows)} ({checked['train']['games']} games), "
        f"val={len(val_rows)} ({checked['validation']['games']} games); "
        f"unsolved episodes dropped={counts['episodes_dropped_unsolved']}"
    )
    print(audit_path)


if __name__ == "__main__":
    main()
