#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.analysis.gradient import (
    assert_shared_teacher_contract,
    teacher_contract_from_manifest,
)
from omniopd.analysis.m3_panel import build_m3_panel
from omniopd.io import read_jsonl, rollout_turn_from_dict, write_jsonl
from omniopd.provenance import (
    fingerprint_code_tree,
    git_revision,
    sha256_file,
    sha256_json,
)


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None or not raw_path:
        raise argparse.ArgumentTypeError("value must contain a safe NAME and non-empty PATH")
    return name, Path(raw_path)


def _load_table(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return list(read_jsonl(path))
    if path.suffix == ".parquet":
        import pandas as pd

        return [
            {key: _python_value(value) for key, value in row.items()}
            for row in pd.read_parquet(path).to_dict(orient="records")
        ]
    raise SystemExit("M3 source data must end in .jsonl or .parquet")


def _python_value(value):
    if isinstance(value, dict):
        return {key: _python_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _python_value(value.tolist())
    return value


def _unique_mapping(values: list[tuple[str, Path]], label: str) -> dict[str, Path]:
    result = dict(values)
    if len(result) != len(values):
        raise SystemExit(f"{label} names must be unique")
    return result


def _resolve_bound_path(parent: Path, declared: str) -> Path:
    path = Path(declared)
    return path if path.is_absolute() else parent.parent / path


def _load_bound_correction_manifest(
    audit: dict,
    audit_path: Path,
    *,
    expected_code: dict,
    expected_revision: str,
) -> tuple[dict, Path]:
    binding = audit.get("correction_manifest")
    if not isinstance(binding, dict):
        raise SystemExit(f"M3 source audit lacks correction manifest: {audit_path}")
    declared_path = binding.get("path")
    declared_sha = binding.get("sha256")
    if (
        not isinstance(declared_path, str)
        or not declared_path
        or not isinstance(declared_sha, str)
        or not declared_sha
    ):
        raise SystemExit(f"M3 source correction binding is incomplete: {audit_path}")
    path = _resolve_bound_path(audit_path, declared_path)
    if not path.is_file() or sha256_file(path) != declared_sha:
        raise SystemExit(f"M3 source correction binding is missing or stale: {audit_path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact") != "teacher_corrections"
        or manifest.get("protocol_version") != "omniopd-v1"
        or manifest.get("code") != expected_code
        or manifest.get("code_revision") != expected_revision
        or manifest.get("corrections_sha256") != audit.get("input_sha256")
        or manifest.get("experiment") != audit.get("experiment")
        or manifest.get("experiment_config", {}).get("sha256")
        != audit.get("experiment_config", {}).get("sha256")
    ):
        raise SystemExit(f"M3 source correction manifest disagrees with audit: {audit_path}")
    return manifest, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the correction-level M3 source/neighbor scoring panel"
    )
    parser.add_argument("--state-pool", required=True)
    parser.add_argument("--state-pool-manifest", required=True)
    parser.add_argument(
        "--source",
        action="append",
        type=_named_path,
        required=True,
        help="A1=ACTUAL_TRAIN_SPLIT and A3=ACTUAL_TRAIN_SPLIT",
    )
    parser.add_argument(
        "--source-audit",
        action="append",
        type=_named_path,
        required=True,
        help="A1=BUILD_AUDIT.json and A3=BUILD_AUDIT.json",
    )
    parser.add_argument(
        "--selection",
        action="append",
        type=_named_path,
        required=True,
        help="NAME=SELECTION.jsonl; include A1/A3 and every design to exclude",
    )
    parser.add_argument(
        "--selection-manifest",
        action="append",
        type=_named_path,
        required=True,
        help="NAME=SELECTION_MANIFEST.json paired one-to-one with --selection",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--minimum-neighbors", type=int, default=5)
    parser.add_argument("--minimum-similarity", type=float, default=0.0)
    parser.add_argument("--bow-weight", type=float, default=0.5)
    args = parser.parse_args()

    state_pool_path = Path(args.state_pool)
    state_pool_manifest_path = Path(args.state_pool_manifest)
    sources = _unique_mapping(args.source, "source")
    source_audits = _unique_mapping(args.source_audit, "source-audit")
    selections = _unique_mapping(args.selection, "selection")
    selection_manifests = _unique_mapping(
        args.selection_manifest, "selection-manifest"
    )
    if set(sources) != {"A1", "A3"} or set(source_audits) != set(sources):
        raise SystemExit("--source and --source-audit names must be exactly A1 and A3")
    if set(selections) != set(selection_manifests) or not {"A1", "A3"} <= set(
        selections
    ):
        raise SystemExit(
            "selection files/manifests must have identical names and include A1/A3"
        )
    all_inputs = [
        state_pool_path,
        state_pool_manifest_path,
        *sources.values(),
        *source_audits.values(),
        *selections.values(),
        *selection_manifests.values(),
    ]
    missing = [str(path) for path in all_inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"M3 preparation inputs are missing: {missing}")
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"refusing to overwrite M3 preparation output: {output}")

    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if not current_revision:
        raise SystemExit("M3 preparation requires an immutable Git revision")
    state_pool_sha256 = sha256_file(state_pool_path)
    state_pool_manifest = json.loads(
        state_pool_manifest_path.read_text(encoding="utf-8")
    )
    if (
        state_pool_manifest.get("artifact") != "immutable_state_pool"
        or state_pool_manifest.get("protocol_version") != "omniopd-v1"
        or state_pool_manifest.get("outputs", {}).get("state_pool_sha256")
        != state_pool_sha256
        or state_pool_manifest.get("code") != current_code
        or state_pool_manifest.get("code_revision") != current_revision
    ):
        raise SystemExit(
            "M3 state pool must match a canonical manifest from this code revision"
        )
    state_pool_manifest_sha256 = sha256_file(state_pool_manifest_path)
    turns = [rollout_turn_from_dict(row) for row in read_jsonl(state_pool_path)]
    selected_hashes: dict[str, set[str]] = {}
    selection_inputs = {}
    selection_experiments = {}
    for name in sorted(selections):
        selection_rows = list(read_jsonl(selections[name]))
        hashes = [str(row["state_hash"]) for row in selection_rows]
        if not hashes or len(hashes) != len(set(hashes)):
            raise SystemExit(f"selection {name} must contain unique non-empty state hashes")
        manifest = json.loads(selection_manifests[name].read_text(encoding="utf-8"))
        expected = {
            "artifact": "selection_manifest",
            "protocol_version": "omniopd-v1",
            "code": current_code,
            "code_revision": current_revision,
            "state_pool_sha256": state_pool_sha256,
            "state_pool_manifest_sha256": state_pool_manifest_sha256,
            "selection_sha256": sha256_file(selections[name]),
            "selected_state_hashes": sorted(hashes),
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise SystemExit(f"selection {name} disagrees with its manifest: {mismatches}")
        experiment = manifest.get("experiment")
        experiment_config_sha256 = manifest.get("experiment_config_sha256")
        if (
            not isinstance(experiment, str)
            or not experiment.strip()
            or not isinstance(experiment_config_sha256, str)
            or not experiment_config_sha256
        ):
            raise SystemExit(
                f"selection {name} lacks an experiment/config identity"
            )
        selected_hashes[name] = set(hashes)
        selection_experiments[name] = experiment.strip()
        selection_inputs[name] = {
            "selection_path": str(selections[name]),
            "selection_sha256": expected["selection_sha256"],
            "manifest_path": str(selection_manifests[name]),
            "manifest_sha256": sha256_file(selection_manifests[name]),
            "experiment": experiment.strip(),
            "experiment_config_sha256": experiment_config_sha256,
            "states": len(hashes),
        }

    source_rows = {name: _load_table(path) for name, path in sources.items()}
    source_input_metadata = {}
    source_experiments = {}
    source_teacher_contracts = {}
    for name in sorted(sources):
        audit = json.loads(source_audits[name].read_text(encoding="utf-8"))
        expected_source_hash = sha256_file(sources[name])
        correction_manifest = audit.get("correction_manifest")
        if not isinstance(correction_manifest, dict):
            raise SystemExit(f"M3 source {name} audit lacks correction manifest")
        correction_manifest_raw_path = correction_manifest.get("path")
        correction_manifest_sha = correction_manifest.get("sha256")
        if (
            not isinstance(correction_manifest_raw_path, str)
            or not correction_manifest_raw_path
            or not isinstance(correction_manifest_sha, str)
            or not correction_manifest_sha
        ):
            raise SystemExit(f"M3 source {name} correction binding is incomplete")
        correction_manifest_path = Path(correction_manifest_raw_path)
        if not correction_manifest_path.is_absolute():
            correction_manifest_path = source_audits[name].parent / correction_manifest_path
        if (
            not correction_manifest_path.is_file()
            or sha256_file(correction_manifest_path) != correction_manifest_sha
        ):
            raise SystemExit(
                f"M3 source {name} bound correction manifest is missing or changed"
            )
        correction = json.loads(
            correction_manifest_path.read_text(encoding="utf-8")
        )
        try:
            source_teacher_contracts[name] = teacher_contract_from_manifest(
                correction
            )
        except ValueError as error:
            raise SystemExit(
                f"M3 source {name} has an invalid Teacher contract: {error}"
            ) from error
        experiment = audit.get("experiment")
        experiment_config = audit.get("experiment_config")
        if (
            audit.get("artifact") != "action_only_training_data_audit"
            or audit.get("training_output_sha256") != expected_source_hash
            or audit.get("protocol_version") != "omniopd-v1"
            or audit.get("code") != current_code
            or audit.get("code_revision") != current_revision
            or audit.get("weighting_mode") != "game_state_mean"
            or not isinstance(audit.get("split"), dict)
            or int(audit.get("validation_rows", 0)) <= 0
            or int(audit.get("training_rows", -1)) != len(source_rows[name])
            or not isinstance(experiment, str)
            or not experiment.strip()
            or experiment != selection_experiments[name]
            or not isinstance(experiment_config, dict)
            or experiment_config.get("sha256")
            != selection_inputs[name]["experiment_config_sha256"]
            or correction.get("artifact") != "teacher_corrections"
            or correction.get("protocol_version") != "omniopd-v1"
            or correction.get("code") != current_code
            or correction.get("code_revision") != current_revision
            or correction.get("experiment") != experiment
            or correction.get("corrections_sha256") != audit.get("input_sha256")
            or int(correction.get("teacher_samples_per_state_N", -1)) != 1
            or correction.get("experiment_config", {}).get("sha256")
            != experiment_config.get("sha256")
        ):
            raise SystemExit(
                f"M3 source {name} must be the actual canonical training split and use N=1"
            )
        state_hashes = [str(row.get("state_hash", "")) for row in source_rows[name]]
        if (
            not all(state_hashes)
            or len(state_hashes) != len(set(state_hashes))
            or any(int(row.get("teacher_sample_index", -1)) != 0 for row in source_rows[name])
        ):
            raise SystemExit(
                f"M3 source {name} must contain exactly one trained Teacher action per state"
            )
        source_experiments[name] = experiment.strip()
        source_input_metadata[name] = {
            "path": str(sources[name]),
            "sha256": expected_source_hash,
            "rows": len(source_rows[name]),
            "audit_path": str(source_audits[name]),
            "audit_sha256": sha256_file(source_audits[name]),
            "experiment": experiment.strip(),
            "experiment_config": experiment_config,
            "data_role": "actual_training_split_examples",
            "split": audit["split"],
            "validation_rows_excluded": int(audit["validation_rows"]),
            "correction_manifest": {
                "path": str(correction_manifest_path),
                "sha256": correction_manifest_sha,
            },
        }
    if len(set(source_experiments.values())) != 2:
        raise SystemExit("M3 A1/A3 must be bound to two distinct experiment identities")
    try:
        assert_shared_teacher_contract(source_teacher_contracts.values())
    except ValueError as error:
        raise SystemExit(
            "M3 A1/A3 panels must share one exact Teacher contract: " + str(error)
        ) from error
    score_rows, metadata, attrition = build_m3_panel(
        turns,
        source_rows,
        selected_hashes,
        experiments_by_group=source_experiments,
        neighbors=args.neighbors,
        minimum_neighbors=args.minimum_neighbors,
        minimum_similarity=args.minimum_similarity,
        bow_weight=args.bow_weight,
    )
    output.mkdir(parents=True)
    metadata_path = output / "m3_sources.jsonl"
    attrition_path = output / "m3_attrition.jsonl"
    write_jsonl(metadata_path, metadata)
    write_jsonl(attrition_path, attrition)
    score_outputs = {}
    for group, rows in score_rows.items():
        path = output / f"score_rows_{group}.jsonl"
        write_jsonl(path, rows)
        score_outputs[group] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
        }
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "m3_frozen_scoring_panel",
        "interpretation": "whole_checkpoint_action_imitation_transfer_not_task_utility",
        "code": current_code,
        "code_revision": current_revision,
        "state_pool": {
            "path": str(state_pool_path),
            "sha256": state_pool_sha256,
            "manifest_path": str(state_pool_manifest_path),
            "manifest_sha256": state_pool_manifest_sha256,
            "states": len(turns),
        },
        "source_inputs": source_input_metadata,
        "source_role": "actual_training_split_examples_only",
        "validation_split_examples_excluded": True,
        "teacher_samples_per_state_N": 1,
        "shared_teacher_contract": source_teacher_contracts["A1"],
        "shared_teacher_contract_sha256": sha256_json(
            source_teacher_contracts["A1"]
        ),
        "experiment_identities": source_experiments,
        "selection_inputs": selection_inputs,
        "excluded_state_set_sha256": sha256_json(sorted(set().union(*selected_hashes.values()))),
        "neighbor_contract": {
            "neighbors_K": args.neighbors,
            "minimum_neighbors": args.minimum_neighbors,
            "minimum_similarity": args.minimum_similarity,
            "similarity": "bow_cosine_and_admissible_jaccard",
            "bow_weight": args.bow_weight,
            "other_game_only": True,
            "same_task_type_only": True,
            "teacher_action_must_be_admissible_at_target": True,
            "student_valid_targets_only": True,
            "all_supplied_selection_states_excluded": True,
            "tie_break": "target_state_hash_ascending",
        },
        "source_metadata_sha256": sha256_file(metadata_path),
        "source_rows_retained": len(metadata),
        "common_retained_game_support": sorted(
            {row["game_id"] for row in metadata if row["group"] == "A1"}
            & {row["game_id"] for row in metadata if row["group"] == "A3"}
        ),
        "common_retained_game_support_required": True,
        "attrition_sha256": sha256_file(attrition_path),
        "source_rows_attrited": len(attrition),
        "score_outputs": score_outputs,
        "analysis_ready": bool(metadata)
        and {row["group"] for row in metadata} == {"A1", "A3"},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
