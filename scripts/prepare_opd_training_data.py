#!/usr/bin/env python3
"""Turn a collected Student trajectory pool into veRL token-OPD training rows.

Standard token-level OPD trains on states the Student itself reached, with the
frozen Teacher scoring the Student's own tokens in exactly the same context.
This script performs that conversion for the frozen reference prompt and writes
a manifest, so a training run can be traced back to the rollout pool, the
prompt file and the code revision that produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from omniopd.io import read_jsonl, rollout_turn_from_dict  # noqa: E402
from omniopd.opd_adapter import build_fixed_pool_opd_prompts  # noqa: E402
from omniopd.prompts import build_reference_prompt  # noqa: E402
from omniopd.provenance import (  # noqa: E402
    fingerprint_code_tree,
    git_revision,
    sha256_file,
    sha256_text,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--prompt-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--selection-policy", default="student_pool_all_turns")
    args = parser.parse_args()
    episodes_path = args.episodes.resolve()
    prompt_path = args.prompt_json.resolve()
    output = args.output.resolve()
    manifest_output = args.manifest_output.resolve()
    if (
        output.exists()
        or manifest_output.exists()
        or output == manifest_output
        or not episodes_path.is_file()
        or not prompt_path.is_file()
    ):
        raise SystemExit("invalid OPD data preparation input or output")

    reference = json.loads(prompt_path.read_text(encoding="utf-8"))
    system_prompt = build_reference_prompt(
        str(reference["instruction"]), reference.get("examples") or []
    )

    turns = []
    episodes = 0
    wins = 0
    for episode in read_jsonl(episodes_path):
        episodes += 1
        wins += int(bool(episode.get("won")))
        for raw_turn in episode.get("turns") or ():
            turns.append(rollout_turn_from_dict(raw_turn))
    if not turns:
        raise SystemExit("the episode pool contains no turns")

    selections = [
        {
            "protocol_version": "omniopd-v1",
            "state_hash": turn.state.state_hash,
            "game_id": turn.state.game_id,
            "turn_index": turn.state.turn_index,
            "selection_policy": args.selection_policy,
        }
        for turn in turns
    ]
    rows = build_fixed_pool_opd_prompts(
        turns, selections, student_system_prompt=system_prompt
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )

    by_family = Counter(turn.state.task_type for turn in turns)
    by_game = Counter(turn.state.game_id for turn in turns)
    manifest = {
        "artifact": "verl_token_opd_training_data",
        "protocol_version": "omniopd-v1",
        "objective": "standard_token_level_on_policy_distillation",
        "teacher_context": "identical_to_student_prefix",
        "selection_policy": args.selection_policy,
        "episodes": {
            "path": str(episodes_path),
            "sha256": sha256_file(episodes_path),
            "count": episodes,
            "solved": wins,
        },
        "prompt": {
            "path": str(prompt_path),
            "sha256": sha256_file(prompt_path),
            "system_prompt_sha256": sha256_text(system_prompt),
        },
        "rows": {
            "path": str(output),
            "count": len(rows),
            "sha256": sha256_file(output),
        },
        "games": len(by_game),
        "turns_by_task_type": dict(sorted(by_family.items())),
        "weighting": "one_over_turns_in_game",
        "code_revision": git_revision(ROOT),
        "code": fingerprint_code_tree(ROOT),
        "script_sha256": sha256_file(Path(__file__)),
    }
    with manifest_output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        f"OPD rows: {len(rows)} from {episodes} episodes "
        f"({wins} solved) across {len(by_game)} games"
    )
    print(manifest_output)


if __name__ == "__main__":
    main()
