from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from omniopd.dataset import audit_acceptance, build_training_rows, split_rows_by_game
from omniopd.io import correction_from_dict, read_jsonl
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file
from omniopd.validation import (
    annotation_member_contract,
    audit_correction_records,
    validate_correction_manifest_against_records,
)


def _require_new(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {path}")


def _write_rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        import pandas as pd

        pd.DataFrame(values).to_parquet(path, index=False)
    elif path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(
                    json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
                )
    else:
        raise SystemExit("data output must end in .parquet or .jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build action-only OPD training data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--correction-manifest", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--weighting",
        choices=["sample_mean", "state_mean", "game_state_mean"],
    )
    parser.add_argument("--audit-output")
    parser.add_argument("--validation-output")
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--split-seed", type=int)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    experiment_path = Path(args.experiment_config).resolve()
    correction_manifest_path = Path(args.correction_manifest).resolve()
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    correction_manifest = json.loads(
        correction_manifest_path.read_text(encoding="utf-8")
    )
    repository_root = Path(__file__).resolve().parents[3]
    current_code = fingerprint_code_tree(repository_root)
    current_revision = git_revision(repository_root)
    if not current_revision:
        raise SystemExit("data building requires an immutable Git revision")
    try:
        configured_weighting = str(experiment["training"]["weighting"])
        configured_train_fraction = float(
            experiment["training"]["nominal_train_fraction"]
        )
        configured_steps = int(experiment["training"]["total_optimizer_steps"])
        configured_split_seed = int(experiment["training"]["data_split_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"experiment training config is incomplete: {error}") from error
    weighting = args.weighting or configured_weighting
    val_fraction = (
        args.val_fraction
        if args.val_fraction is not None
        else 1.0 - configured_train_fraction
    )
    split_seed = (
        configured_split_seed if args.split_seed is None else args.split_seed
    )
    if weighting != configured_weighting or abs(
        val_fraction - (1.0 - configured_train_fraction)
    ) >= 1e-12:
        raise SystemExit("data-building overrides disagree with the experiment config")
    if split_seed != configured_split_seed:
        raise SystemExit("--split-seed disagrees with the experiment config")
    manifest_config = correction_manifest.get("experiment_config", {})
    expected_manifest_fields = {
        "artifact": "teacher_corrections",
        "protocol_version": "omniopd-v1",
        "code": current_code,
        "code_revision": current_revision,
        "corrections_sha256": sha256_file(input_path),
        "experiment": experiment.get("experiment"),
    }
    mismatches = {
        key: (correction_manifest.get(key), expected)
        for key, expected in expected_manifest_fields.items()
        if correction_manifest.get(key) != expected
    }
    if manifest_config.get("sha256") != sha256_file(experiment_path):
        mismatches["experiment_config.sha256"] = (
            manifest_config.get("sha256"),
            sha256_file(experiment_path),
        )
    if mismatches:
        raise SystemExit(f"correction artifact is not bound to this experiment: {mismatches}")

    records = [correction_from_dict(record) for record in read_jsonl(input_path)]
    issues = audit_correction_records(records)
    if issues:
        preview = [f"{issue.code}:{issue.message}" for issue in issues[:10]]
        raise SystemExit(f"correction protocol audit failed: {preview}")
    try:
        annotation_contract = validate_correction_manifest_against_records(
            correction_manifest, records
        )
    except ValueError as error:
        raise SystemExit(f"correction manifest contradicts its records: {error}") from error
    state_hashes = [record.state.state_hash for record in records]
    if len(state_hashes) != len(set(state_hashes)):
        raise SystemExit(
            "duplicate state records detected; put all N Teacher draws in one correction record"
        )
    versions = {record.protocol_version for record in records}
    if len(versions) != 1:
        raise SystemExit(f"mixed protocol versions are not allowed: {sorted(versions)}")
    training_rows = build_training_rows(records, weighting=weighting)
    split_metadata = None
    if args.validation_output:
        train_rows, validation_rows, split_metadata = split_rows_by_game(
            training_rows, val_fraction=val_fraction, rng_seed=split_seed
        )
    else:
        train_rows, validation_rows = training_rows, []
    rows = [row.to_dict() for row in train_rows]
    if not rows:
        raise SystemExit("no valid Teacher samples were available")
    output = Path(args.output).resolve()
    audit_path = Path(args.audit_output or str(output) + ".audit.json").resolve()
    destinations = [output, audit_path]
    if args.validation_output:
        destinations.append(Path(args.validation_output).resolve())
    if len(destinations) != len(set(destinations)):
        raise SystemExit("training, validation, and audit outputs must be distinct paths")
    for destination in destinations:
        _require_new(destination)
    _write_rows(output, rows)
    if args.validation_output:
        validation_output = Path(args.validation_output).resolve()
        values = [row.to_dict() for row in validation_rows]
        _write_rows(validation_output, values)
    report = audit_acceptance(records)
    report["artifact"] = "action_only_training_data_audit"
    report["code"] = current_code
    report["code_revision"] = current_revision
    report["split"] = split_metadata
    report["input_sha256"] = sha256_file(input_path)
    report["training_output_sha256"] = sha256_file(output)
    report["validation_output_sha256"] = (
        sha256_file(Path(args.validation_output).resolve())
        if args.validation_output
        else None
    )
    report["protocol_version"] = next(iter(versions))
    report["weighting_mode"] = weighting
    report["experiment"] = experiment["experiment"]
    report["experiment_config"] = {
        "path": str(experiment_path),
        "sha256": sha256_file(experiment_path),
    }
    report["correction_manifest"] = {
        "path": str(correction_manifest_path),
        "sha256": sha256_file(correction_manifest_path),
    }
    report["annotation_contract"] = {
        **annotation_contract,
        "member": annotation_member_contract(
            correction_manifest,
            annotation_manifest_sha256=sha256_file(correction_manifest_path),
        ),
    }
    report["training_contract"] = {
        "global_batch_size": int(experiment["training"]["global_batch_size"]),
        "total_optimizer_steps": configured_steps,
        "sampling_unit": experiment["training"]["sampling_unit"],
        "weighting": weighting,
        "nominal_train_fraction": configured_train_fraction,
        "split_unit": "game",
        "split_seed": split_seed,
    }
    report["training_rows"] = len(train_rows)
    report["validation_rows"] = len(validation_rows)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
