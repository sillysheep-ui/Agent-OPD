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
    annotation_member_contract,
    annotation_shared_contract,
    compare_control_protocols,
    validate_annotation_run_manifests,
    validate_fixed_budget_arms,
    validate_training_run_manifests,
)
from omniopd.evaluation import (
    validate_aggregate_pair_contracts,
    validate_annotation_pair_manifest,
)
from omniopd.provenance import sha256_file, sha256_json


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "kind": "file",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _annotation_member(path: Path, manifest: dict) -> dict[str, object]:
    return annotation_member_contract(
        manifest, annotation_manifest_sha256=sha256_file(path)
    )


def build_annotation_pair_manifest(paths: list[Path], manifests: list[dict]) -> dict:
    """Build the immutable bridge from realized annotation arms to training."""

    if len(paths) != 2 or len(manifests) != 2:
        raise ValueError("confirmatory annotation pair requires exactly two manifest files")
    for path, manifest in zip(paths, manifests):
        if not path.is_file():
            raise ValueError(f"annotation manifest file does not exist: {path}")
        try:
            on_disk = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read annotation manifest {path}: {error}") from error
        if on_disk != manifest:
            raise ValueError("annotation manifest object does not reproduce its on-disk bytes")
    validate_annotation_run_manifests(manifests)
    members = [_annotation_member(path, manifest) for path, manifest in zip(paths, manifests)]
    experiments = [str(member["experiment"]) for member in members]
    if len(set(experiments)) != len(experiments):
        raise ValueError("annotation pair experiments must be unique")
    if len(members) != 2:
        raise ValueError("confirmatory annotation pair must contain exactly two arms")
    left, right = members
    if left["state_pool_manifest_sha256"] != right["state_pool_manifest_sha256"]:
        raise ValueError("annotation arms did not use the same state-pool manifest")
    shared_contracts = [annotation_shared_contract(manifest) for manifest in manifests]
    if shared_contracts[0] != shared_contracts[1]:
        raise ValueError(
            "annotation arms do not share the same complete Teacher/state-pool contract"
        )
    if (
        left["distinct_states_M"] == right["distinct_states_M"]
        or left["teacher_samples_per_state_N"] == right["teacher_samples_per_state_N"]
        or (int(left["distinct_states_M"]) - int(right["distinct_states_M"]))
        * (
            int(left["teacher_samples_per_state_N"])
            - int(right["teacher_samples_per_state_N"])
        )
        >= 0
    ):
        raise ValueError("annotation pair is not a breadth/depth trade-off")
    breadth = max(members, key=lambda item: int(item["distinct_states_M"]))
    depth = min(members, key=lambda item: int(item["distinct_states_M"]))
    pair_contract = {
        "comparison_contrast": "breadth_depth",
        "effect_direction": "breadth_minus_depth",
        "arm_roles": {
            "breadth": breadth["experiment"],
            "depth": depth["experiment"],
        },
        "shared_contract": shared_contracts[0],
        "members": {str(member["experiment"]): member for member in members},
    }
    payload = {
        "protocol_version": "omniopd-v1",
        "artifact": "omniopd_annotation_pair",
        "schema_version": 2,
        "pair_kind": "fixed_budget_annotation_runs",
        "comparison_contrast": "breadth_depth",
        "annotation_manifests": [
            {"experiment": member["experiment"], **_file_identity(path)}
            for path, member in zip(paths, members)
        ],
        "pair_contract": pair_contract,
        "pair_contract_sha256": sha256_json(pair_contract),
    }
    validate_annotation_pair_manifest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a preregistered experiment pair")
    parser.add_argument(
        "--kind",
        choices=[
            "fixed_budget",
            "state_source_control",
            "annotation_runs",
            "training_runs",
            "evaluation_aggregates",
        ],
        required=True,
    )
    parser.add_argument("configs", nargs=2)
    parser.add_argument(
        "--output",
        help="write the content-addressed pair manifest (required for annotation_runs)",
    )
    args = parser.parse_args()
    if args.kind in {"annotation_runs", "training_runs", "evaluation_aggregates"}:
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
        if any(
            config.get("artifact_status") != "design_constraint_template"
            or config.get("confirmatory_use_allowed") is not False
            for config in configs
        ):
            raise SystemExit("state-source controls must be marked as non-runnable templates")
        report = {
            "design_constraint_valid": True,
            "confirmatory_pipeline_implemented": False,
            "kind": args.kind,
            "allowed_intervention_fields": [
                "state_source.name",
                "state_source.behavior_policy",
            ],
            "limitation": "not accepted by the canonical fixed-budget training launcher",
        }
    elif args.kind == "annotation_runs":
        if not args.output:
            raise SystemExit("--output is required for annotation_runs")
        output = Path(args.output)
        if output.exists():
            raise SystemExit("refusing to overwrite annotation pair manifest")
        report = build_annotation_pair_manifest(
            [Path(path) for path in args.configs], configs
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    elif args.kind == "training_runs":
        validate_training_run_manifests(configs)
        report = {"valid": True, "kind": args.kind, "shared": "training_contract"}
    else:
        pairing = validate_aggregate_pair_contracts(configs[0], configs[1])
        report = {
            "metadata_contract_valid": True,
            "full_data_validation_performed": False,
            "kind": args.kind,
            "comparison_contrast": pairing["comparison_contrast"],
            "treatment_experiment": pairing["treatment_experiment"],
            "control_experiment": pairing["control_experiment"],
            "shared": "paired games/seeds and evaluation protocol",
            "next_required_step": (
                "run omniopd-bootstrap with both aggregate data files and manifests"
            ),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
