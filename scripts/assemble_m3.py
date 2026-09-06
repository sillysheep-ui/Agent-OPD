#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.analysis.m3_panel import M3_GROUPS, assemble_m3_rows
from omniopd.io import read_jsonl, write_jsonl
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must be NAME=PATH")
    name, path = value.split("=", 1)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None or not path:
        raise argparse.ArgumentTypeError("value must be NAME=PATH")
    return name, Path(path)


def _score_cell(value: str) -> tuple[tuple[str, str], Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("cell must be CHECKPOINT,PANEL=PATH")
    identity, path = value.split("=", 1)
    parts = identity.split(",")
    if len(parts) != 2 or any(part not in M3_GROUPS for part in parts) or not path:
        raise argparse.ArgumentTypeError("cell must be A1|A3,A1|A3=PATH")
    return (parts[0], parts[1]), Path(path)


def _identity(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        key: _identity(item)
        for key, item in value.items()
        if key not in {"path", "resolved_path", "value"}
    }


def _unique(values, label):
    result = dict(values)
    if len(result) != len(values):
        raise SystemExit(f"{label} identities must be unique")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join the required M3 checkpoint×panel 2x2 score grid"
    )
    parser.add_argument("--preparation-manifest", required=True)
    parser.add_argument("--base-score", action="append", type=_named_path, required=True)
    parser.add_argument(
        "--updated-score", action="append", type=_score_cell, required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output")
    args = parser.parse_args()

    base_paths = _unique(args.base_score, "base-score")
    updated_paths = _unique(args.updated_score, "updated-score")
    expected_cells = {
        (checkpoint, panel) for checkpoint in M3_GROUPS for panel in M3_GROUPS
    }
    if set(base_paths) != M3_GROUPS or set(updated_paths) != expected_cells:
        raise SystemExit(
            "M3 requires BASE×{A1,A3} and the complete {A1,A3}×{A1,A3} grid"
        )
    preparation_path = Path(args.preparation_manifest)
    output = Path(args.output)
    manifest_output = Path(args.manifest_output or str(output) + ".manifest.json")
    score_paths = [*base_paths.values(), *updated_paths.values()]
    score_manifest_paths = [Path(str(path) + ".manifest.json") for path in score_paths]
    if any(
        not path.is_file()
        for path in [preparation_path, *score_paths, *score_manifest_paths]
    ):
        raise SystemExit("preparation, every score, and every score manifest must exist")
    if output == manifest_output or output.exists() or manifest_output.exists():
        raise SystemExit("refusing to overwrite or alias M3 assembled outputs")

    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    experiments = preparation.get("experiment_identities")
    if (
        preparation.get("artifact") != "m3_frozen_scoring_panel"
        or preparation.get("analysis_ready") is not True
        or not isinstance(experiments, dict)
        or set(experiments) != M3_GROUPS
        or len(set(experiments.values())) != 2
    ):
        raise SystemExit("preparation manifest lacks analysis-ready A1/A3 identities")
    metadata_path = preparation_path.parent / "m3_sources.jsonl"
    if not metadata_path.is_file() or sha256_file(metadata_path) != preparation.get(
        "source_metadata_sha256"
    ):
        raise SystemExit("M3 source metadata does not match its preparation manifest")

    common_identity = None
    comparison_contract = None
    checkpoint_identities: dict[str, str] = {}
    manifests = {}
    base_rows = {}
    updated_rows = {}
    entries = [
        ("BASE", panel, path) for panel, path in sorted(base_paths.items())
    ] + [
        (checkpoint, panel, path)
        for (checkpoint, panel), path in sorted(updated_paths.items())
    ]
    for checkpoint, panel, path in entries:
        manifest_path = Path(str(path) + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_data_hash = preparation["score_outputs"][panel]["sha256"]
        if (
            manifest.get("artifact") != "action_imitation_log_probability"
            or manifest.get("scores_sha256") != sha256_file(path)
            or manifest.get("data_sha256") != expected_data_hash
            or manifest.get("checkpoint_group") != checkpoint
            or manifest.get("panel_group") != panel
            or manifest.get("panel_experiment") != experiments[panel]
            or manifest.get("token_contract")
            != "assistant_action_content_tokens_v1"
            or manifest.get("enable_thinking") is not False
        ):
            raise SystemExit(f"M3 score manifest identity mismatch at {checkpoint}×{panel}")
        identity = json.dumps(
            {
                "code": manifest.get("code"),
                "code_revision": manifest.get("code_revision"),
                "tokenizer": _identity(manifest.get("tokenizer")),
                "base_model": _identity(manifest.get("base_model")),
                "model_dtype": manifest.get("model_dtype"),
            },
            sort_keys=True,
            allow_nan=False,
        )
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise SystemExit("all six M3 score cells must share code/base/tokenizer/dtype")
        adapter = manifest.get("adapter")
        training = manifest.get("training_manifest")
        if checkpoint == "BASE":
            if adapter is not None or training is not None or manifest.get(
                "checkpoint_experiment"
            ) is not None:
                raise SystemExit("M3 BASE cells cannot load or claim a training run")
            base_rows[panel] = list(read_jsonl(path))
        else:
            if (
                not isinstance(adapter, dict)
                or not isinstance(training, dict)
                or training.get("final_checkpoint_verified") is not True
                or training.get("experiment") != experiments[checkpoint]
                or manifest.get("checkpoint_experiment") != experiments[checkpoint]
                or not training.get("completion_sha256")
                or not isinstance(training.get("comparison_contract"), dict)
            ):
                raise SystemExit(
                    f"M3 {checkpoint} checkpoint lacks a completed matching experiment"
                )
            run_identity = json.dumps(
                {
                    "adapter": _identity(adapter),
                    "launch": training.get("sha256"),
                    "completion": training.get("completion_sha256"),
                    "final_checkpoint": training.get("final_checkpoint"),
                },
                sort_keys=True,
                allow_nan=False,
            )
            previous = checkpoint_identities.setdefault(checkpoint, run_identity)
            if previous != run_identity:
                raise SystemExit(
                    f"M3 {checkpoint} rows were scored by different checkpoint bytes"
                )
            contract = json.dumps(
                training["comparison_contract"], sort_keys=True, allow_nan=False
            )
            if comparison_contract is None:
                comparison_contract = contract
            elif comparison_contract != contract:
                raise SystemExit(
                    "A1/A3 checkpoints must share seed, steps, and key hyperparameters"
                )
            updated_rows[(checkpoint, panel)] = list(read_jsonl(path))
        manifests[(checkpoint, panel)] = (path, manifest_path, manifest)
    if len(set(checkpoint_identities.values())) != 2:
        raise SystemExit("A1 and A3 must be distinct completed checkpoint artifacts")

    current_identity = json.dumps(
        {"code": fingerprint_code_tree(ROOT), "code_revision": git_revision(ROOT)},
        sort_keys=True,
        allow_nan=False,
    )
    preparation_identity = json.dumps(
        {
            "code": preparation.get("code"),
            "code_revision": preparation.get("code_revision"),
        },
        sort_keys=True,
        allow_nan=False,
    )
    score_code_identity = json.dumps(
        {
            "code": manifests[("BASE", "A1")][2].get("code"),
            "code_revision": manifests[("BASE", "A1")][2].get("code_revision"),
        },
        sort_keys=True,
        allow_nan=False,
    )
    if current_identity != preparation_identity or current_identity != score_code_identity:
        raise SystemExit("M3 preparation, scoring, and assembly require one code revision")

    metadata = list(read_jsonl(metadata_path))
    assembled = assemble_m3_rows(metadata, base_rows, updated_rows)
    write_jsonl(output, assembled)
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "m3_crossed_correction_level_transfer_rows",
        "interpretation": "crossed_whole_checkpoint_action_imitation_transfer_not_task_utility",
        "code": fingerprint_code_tree(ROOT),
        "code_revision": git_revision(ROOT),
        "preparation_manifest": {
            "path": str(preparation_path),
            "sha256": sha256_file(preparation_path),
        },
        "experiment_identities": experiments,
        "score_grid": {
            f"{checkpoint}x{panel}": {
                "path": str(path),
                "sha256": sha256_file(path),
                "manifest_sha256": sha256_file(manifest_path),
            }
            for (checkpoint, panel), (path, manifest_path, _) in manifests.items()
        },
        "comparison_contract": json.loads(comparison_contract),
        "design": "checkpoint_by_panel_2x2_with_shared_base_per_panel",
        "token_contract": "assistant_action_content_tokens_v1",
        "rows": len(assembled),
        "cells": {
            f"{checkpoint}x{panel}": sum(
                row["checkpoint_group"] == checkpoint and row["panel_group"] == panel
                for row in assembled
            )
            for checkpoint, panel in expected_cells
        },
        "output_sha256": sha256_file(output),
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
