#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.validation import (
    compare_control_protocols,
    validate_annotation_run_manifests,
    validate_fixed_budget_arms,
    validate_training_run_manifests,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a preregistered experiment pair")
    parser.add_argument(
        "--kind",
        choices=[
            "fixed_budget",
            "state_source_control",
            "annotation_runs",
            "training_runs",
        ],
        required=True,
    )
    parser.add_argument("configs", nargs=2)
    args = parser.parse_args()
    if args.kind in {"annotation_runs", "training_runs"}:
        configs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.configs]
    else:
        configs = [yaml.safe_load(Path(path).read_text(encoding="utf-8")) for path in args.configs]
    if args.kind == "fixed_budget":
        validate_fixed_budget_arms(configs)
        report = {
            "valid": True,
            "kind": args.kind,
            "allowed_differences": [
                "experiment",
                "states_per_game",
                "distinct_states_M",
                "teacher_samples_per_state_N",
            ],
        }
    elif args.kind == "state_source_control":
        differences = compare_control_protocols(
            configs[0],
            configs[1],
            allowed_differences=frozenset(
                {
                    "experiment",
                    "state_source.name",
                    "state_source.behavior_policy",
                }
            ),
        )
        if differences:
            raise SystemExit(f"control has non-state-source differences: {differences}")
        report = {
            "valid": True,
            "kind": args.kind,
            "allowed_intervention_fields": [
                "state_source.name",
                "state_source.behavior_policy",
            ],
        }
    elif args.kind == "annotation_runs":
        validate_annotation_run_manifests(configs)
        report = {"valid": True, "kind": args.kind, "shared": "B,pool,q_T,selection"}
    else:
        validate_training_run_manifests(configs)
        report = {"valid": True, "kind": args.kind, "shared": "training_contract"}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
