#!/usr/bin/env python3
"""Read-only audit of a nonconfirmatory paired ALFWorld baseline pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    arms = ("qwen3_14b", "qwen3_4b")
    episodes = {arm: _read_jsonl(root / arm / "episodes.jsonl") for arm in arms}
    ledgers = {arm: _read_jsonl(root / arm / "requests.jsonl") for arm in arms}
    identities = [
        [(row["game_id"], row["game_sha256"], row["environment_seed"]) for row in episodes[arm]]
        for arm in arms
    ]
    print("same_game_file_and_environment_seed:", identities[0] == identities[1])
    for arm in arms:
        print("\nMODEL", arm)
        print("request_statuses:", dict(Counter(row["status"] for row in ledgers[arm])))
        print("episodes:", len(episodes[arm]), "requests:", len(ledgers[arm]))
        for episode in episodes[arm]:
            turns = episode["turns"]
            actions = [turn["student"]["executed_action"] for turn in turns]
            dropped = sum(
                int(turn["state"]["truncation"]["dropped_pairs"]) > 0
                for turn in turns
            )
            print(
                episode["task_type"],
                "won=", episode["won"],
                "steps=", episode["steps"],
                "dropped_history_turns=", dropped,
                "top_actions=", Counter(actions).most_common(3),
                "last_five=", actions[-5:],
            )


if __name__ == "__main__":
    main()
