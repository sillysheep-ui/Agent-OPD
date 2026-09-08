#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.evaluation import aggregate_evaluation_results
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file


def _parse_evaluation(value: str) -> tuple[int, Path]:
    if re.fullmatch(r"[0-9]+=.+", value) is None:
        raise argparse.ArgumentTypeError("evaluation must be SEED=RESULT.json")
    seed, path = value.split("=", 1)
    return int(seed), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluated checkpoints into seed-to-game bootstrap input"
    )
    parser.add_argument(
        "--evaluation", action="append", type=_parse_evaluation, required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output")
    args = parser.parse_args()

    output = Path(args.output)
    manifest_path = Path(args.manifest_output or str(output) + ".manifest.json")
    if output == manifest_path or output.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite aggregate output or manifest")
    seeds = [seed for seed, _ in args.evaluation]
    if len(seeds) != len(set(seeds)):
        raise SystemExit("evaluation seeds must be unique")
    if any(not path.exists() for _, path in args.evaluation):
        raise SystemExit("every evaluation result must exist")
    evaluations = {
        seed: json.loads(path.read_text(encoding="utf-8"))
        for seed, path in args.evaluation
    }
    bundle, contract = aggregate_evaluation_results(evaluations)
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if contract.get("code") != current_code or contract.get("code_revision") != current_revision:
        raise SystemExit(
            "evaluation and aggregation must use the same immutable code revision"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "hierarchical_bootstrap_input",
        "schema_version": 2,
        "code": current_code,
        "code_revision": current_revision,
        "experiment": contract["experiment"],
        "arm_contract": contract["arm_contract"],
        "evaluation_inputs": [
            {"seed": seed, "path": str(path), "sha256": sha256_file(path)}
            for seed, path in sorted(args.evaluation)
        ],
        "evaluation_contract": contract,
        "training_runs": {
            str(seed): {
                "experiment": evaluations[seed]["experiment"],
                "training_seed": seed,
                "completion_status": evaluations[seed]["training_completion"][
                    "completion_status"
                ],
                "final_global_step": evaluations[seed]["training_completion"][
                    "final_global_step"
                ],
                "checkpoint_artifact": evaluations[seed]["checkpoint_artifact"],
                "training_launch_manifest_sha256": evaluations[seed][
                    "training_launch_manifest_sha256"
                ],
                "training_completion_manifest_sha256": evaluations[seed][
                    "training_completion_manifest_sha256"
                ],
                "service_manifest_sha256": evaluations[seed][
                    "service_manifest_sha256"
                ],
                "annotation_pair_contract_sha256": evaluations[seed][
                    "training_completion"
                ]["annotation_pair_binding"]["pair_contract_sha256"],
            }
            for seed in sorted(evaluations)
        },
        "seeds": sorted(bundle),
        "games": len(next(iter(bundle.values()))),
        "game_ids": sorted(next(iter(bundle.values()))),
        "output_sha256": sha256_file(output),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
