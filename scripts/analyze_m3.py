#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.analysis.m3_stats import fit_transfer_regression, game_cluster_bootstrap_coefficient
from omniopd.io import read_jsonl
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Correction-level M3 action-transfer analysis")
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    input_path = Path(args.input)
    input_manifest_path = Path(args.input_manifest)
    output = Path(args.output)
    if not input_path.is_file() or not input_manifest_path.is_file():
        raise SystemExit("M3 rows and their manifest must exist")
    if output.exists():
        raise SystemExit(f"refusing to overwrite M3 analysis: {output}")
    if args.replicates <= 0:
        raise SystemExit("--replicates must be positive")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if (
        input_manifest.get("artifact")
        != "m3_crossed_correction_level_transfer_rows"
        or input_manifest.get("output_sha256") != sha256_file(input_path)
        or input_manifest.get("token_contract") != "assistant_action_content_tokens_v1"
        or input_manifest.get("code") != current_code
        or input_manifest.get("code_revision") != current_revision
    ):
        raise SystemExit("M3 rows do not match a canonical assembled-input manifest")
    rows = list(read_jsonl(args.input))
    result = fit_transfer_regression(rows)
    intervals = {}
    for coefficient in [
        "checkpoint_A3",
        "S_x_checkpoint_A3",
        "checkpoint_x_panel_A3",
        "S_x_checkpoint_x_panel_A3",
    ]:
        point, lower, upper = game_cluster_bootstrap_coefficient(
            rows,
            coefficient=coefficient,
            replicates=args.replicates,
            rng_seed=args.seed,
        )
        intervals[coefficient] = {"estimate": point, "lower_95": lower, "upper_95": upper}
    report = {
        "protocol_version": "omniopd-v1",
        "artifact": "m3_crossed_action_imitation_transfer_analysis",
        "interpretation": "crossed_whole_checkpoint_action_imitation_transfer_not_task_utility",
        "code": current_code,
        "code_revision": current_revision,
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "manifest_path": str(input_manifest_path),
            "manifest_sha256": sha256_file(input_manifest_path),
        },
        "bootstrap_replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "regression": asdict(result),
        "cluster_bootstrap": intervals,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
