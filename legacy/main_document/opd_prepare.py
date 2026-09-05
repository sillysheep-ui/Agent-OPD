#!/usr/bin/env python3
"""Convert Route-A collector JSONL into veRL SFT parquet files.
 
Each valid Teacher sample becomes one training row. With fixed N per selected turn,
this is exactly the empirical multi-sample cross-entropy weighting: repeated Teacher
actions naturally receive repeated weight.
 
IMPORTANT: use final_turn_dataset.py during veRL training, otherwise the default
MultiTurnSFTDataset will also train on historical Student assistant messages.
"""
from __future__ import annotations
 
import argparse
import json
import random
from pathlib import Path
 
import pandas as pd
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--val-output", required=True)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deduplicate-with-weight", action="store_true",
                   help="Reserved for custom weighted loss; do not use with vanilla SFT trainer")
    return p.parse_args()
 
 
def main() -> None:
    args = parse_args()
    if args.deduplicate_with_weight:
        raise NotImplementedError("Vanilla veRL SFT does not consume per-row weights; keep repeated MC samples instead.")
 
    rows = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            context = rec["turn"]["query_messages"]
            for sample in rec.get("teacher_samples", []):
                if not sample.get("valid") or not sample.get("target_response"):
                    continue
                messages = list(context) + [{"role": "assistant", "content": sample["target_response"]}]
                rows.append({
                    "messages": messages,
                    "enable_thinking": False,
                    "episode_index": rec.get("episode_index"),
                    "gamefile": rec.get("gamefile"),
                    "turn_index": rec["turn"].get("turn_index"),
                    "teacher_action": sample.get("canonical_action"),
                    "teacher_sample_index": sample.get("sample_index"),
                })
 
    if not rows:
        raise RuntimeError("No valid Teacher samples found; inspect collector parsing/Teacher prompt before training.")
 
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(round(len(rows) * args.val_fraction))) if len(rows) > 1 else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:] if n_val else rows
    if not train_rows:
        train_rows, val_rows = rows, []
 
    train_path = Path(args.train_output)
    val_path = Path(args.val_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
 
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    if val_rows:
        pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    else:
        pd.DataFrame(train_rows[:1]).to_parquet(val_path, index=False)
 
    print(json.dumps({
        "raw_valid_samples": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows) if val_rows else 1,
        "train_output": str(train_path),
        "val_output": str(val_path),
    }, indent=2))
 
 
if __name__ == "__main__":
    main()
