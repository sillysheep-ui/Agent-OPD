#!/usr/bin/env python3
"""Write one nonconfirmatory veRL OPD prompt from a real ALFWorld state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.io import read_jsonl, rollout_turn_from_dict
from omniopd.opd_adapter import build_fixed_pool_opd_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-pool", type=Path, required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.state_index < 0 or args.output.exists() or not args.state_pool.is_file():
        raise SystemExit("invalid OPD training smoke input or output")
    turn = next(
        (rollout_turn_from_dict(row) for index, row in enumerate(read_jsonl(args.state_pool))
         if index == args.state_index),
        None,
    )
    if turn is None:
        raise SystemExit("requested ALFWorld state is absent")
    selection = {
        "protocol_version": "omniopd-v1",
        "state_hash": turn.state.state_hash,
        "game_id": turn.state.game_id,
        "turn_index": turn.state.turn_index,
        "selection_policy": "technical_smoke_only",
    }
    row = build_fixed_pool_opd_prompts([turn], [selection])[0]
    row["extra_info"]["nonconfirmatory_smoke"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    print(f"nonconfirmatory OPD row: {turn.state.state_hash}; {args.output}")


if __name__ == "__main__":
    main()
