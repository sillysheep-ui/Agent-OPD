#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import list_alfworld_games
from omniopd.io import read_jsonl, record_game_id
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file
from omniopd.sampling import stable_shuffled


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze an immutable ALFWorld game list")
    parser.add_argument("--env-config", required=True)
    parser.add_argument(
        "--split",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
        required=True,
    )
    parser.add_argument("--limit-games", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.limit_games <= 0 or args.seed < 0:
        raise SystemExit("--limit-games must be positive and --seed non-negative")
    output = Path(args.output)
    manifest_output = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_output.exists():
        raise SystemExit("refusing to overwrite a frozen game list or its manifest")
    if not Path(args.env_config).is_file() or any(
        not Path(path).is_file() for path in args.exclude_jsonl
    ):
        raise SystemExit("environment config and all exclusion inputs must exist")
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if not current_revision:
        raise SystemExit("freezing a game list requires an immutable Git revision")

    config = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    excluded: set[str] = set()
    for path in args.exclude_jsonl:
        for row in read_jsonl(path):
            game = record_game_id(row)
            if game is not None:
                excluded.add(game)
    games = stable_shuffled(
        (
            game
            for game in list_alfworld_games(config, args.split)
            if game not in excluded
        ),
        seed=args.seed,
    )
    games = [str(game) for game in games[: args.limit_games]]
    if len(games) != args.limit_games or len(games) != len(set(games)):
        raise SystemExit("could not construct the requested unique frozen game list")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(games, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "frozen_game_list",
        "code": current_code,
        "code_revision": current_revision,
        "split": args.split,
        "games": len(games),
        "seed": args.seed,
        "excluded_games": len(excluded),
        "env_config_sha256": sha256_file(args.env_config),
        "exclusion_jsonl_sha256": {
            str(path): sha256_file(path) for path in args.exclude_jsonl
        },
        "game_list_sha256": sha256_file(output),
    }
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
