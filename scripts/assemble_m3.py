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

from omniopd.analysis.m3_panel import assemble_m3_rows
from omniopd.io import read_jsonl, write_jsonl
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file


def _score_pair(value: str) -> tuple[str, tuple[Path, Path]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("score pair must be GROUP=BASE.jsonl,UPDATED.jsonl")
    group, paths = value.split("=", 1)
    parts = paths.split(",")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", group) is None or len(parts) != 2:
        raise argparse.ArgumentTypeError("score pair must be GROUP=BASE.jsonl,UPDATED.jsonl")
    return group, (Path(parts[0]), Path(parts[1]))


def _identity(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        key: _identity(item)
        for key, item in value.items()
        if key not in {"path", "resolved_path", "value"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join frozen M3 panel scores into correction-level S_i/T_i rows"
    )
    parser.add_argument("--preparation-manifest", required=True)
    parser.add_argument("--score-pair", action="append", type=_score_pair, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output")
    args = parser.parse_args()

    pairs = dict(args.score_pair)
    if len(pairs) != len(args.score_pair) or set(pairs) != {"A1", "A3"}:
        raise SystemExit("score pairs must contain unique A1 and A3 groups")
    preparation_path = Path(args.preparation_manifest)
    output = Path(args.output)
    manifest_output = Path(args.manifest_output or str(output) + ".manifest.json")
    all_score_paths = [path for pair in pairs.values() for path in pair]
    all_manifest_paths = [Path(str(path) + ".manifest.json") for path in all_score_paths]
    inputs = [preparation_path, *all_score_paths, *all_manifest_paths]
    if any(not path.is_file() for path in inputs):
        raise SystemExit("preparation, score, and score-manifest inputs must all exist")
    if output == manifest_output or output.exists() or manifest_output.exists():
        raise SystemExit("refusing to overwrite or alias M3 assembled outputs")

    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if (
        preparation.get("artifact") != "m3_frozen_scoring_panel"
        or preparation.get("analysis_ready") is not True
        or not preparation.get("code_revision")
    ):
        raise SystemExit("preparation manifest is not an analysis-ready frozen M3 panel")
    metadata_path = preparation_path.parent / "m3_sources.jsonl"
    if (
        not metadata_path.is_file()
        or sha256_file(metadata_path) != preparation.get("source_metadata_sha256")
    ):
        raise SystemExit("M3 source metadata does not match its preparation manifest")

    score_manifests: dict[str, tuple[dict, dict]] = {}
    score_rows = {}
    common_code = None
    common_tokenizer = None
    common_base_model = None
    common_model_dtype = None
    updated_adapters = []
    updated_training_manifests = []
    for group in ["A1", "A3"]:
        group_manifests = tuple(
            json.loads(Path(str(path) + ".manifest.json").read_text(encoding="utf-8"))
            for path in pairs[group]
        )
        expected_data_hash = preparation["score_outputs"][group]["sha256"]
        for role, path, manifest in zip(["base", "updated"], pairs[group], group_manifests):
            if manifest.get("artifact") != "action_imitation_log_probability":
                raise SystemExit(f"{group} {role} manifest has the wrong artifact type")
            if manifest.get("data_sha256") != expected_data_hash:
                raise SystemExit(f"{group} {role} scored a different frozen panel")
            if manifest.get("scores_sha256") != sha256_file(path):
                raise SystemExit(f"{group} {role} score file does not match its manifest")
            if (
                manifest.get("token_contract") != "assistant_action_content_tokens_v1"
                or manifest.get("enable_thinking") is not False
            ):
                raise SystemExit(f"{group} {role} used a non-canonical token contract")
            if not manifest.get("code_revision"):
                raise SystemExit(f"{group} {role} score manifest lacks a code revision")
            code_identity = json.dumps(
                [manifest.get("code"), manifest.get("code_revision")],
                sort_keys=True,
                allow_nan=False,
            )
            tokenizer_identity = json.dumps(
                _identity(manifest.get("tokenizer")), sort_keys=True, allow_nan=False
            )
            base_model_identity = json.dumps(
                _identity(manifest.get("base_model")),
                sort_keys=True,
                allow_nan=False,
            )
            model_dtype = manifest.get("model_dtype")
            adapter_identity = (
                None
                if manifest.get("adapter") is None
                else json.dumps(
                    _identity(manifest.get("adapter")),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            training_manifest_identity = manifest.get("training_manifest")
            if role == "base" and adapter_identity is not None:
                raise SystemExit(f"{group} base scores must not load an adapter")
            if role == "base" and training_manifest_identity is not None:
                raise SystemExit(f"{group} base scores must not name a training run")
            if role == "updated" and (
                adapter_identity is None
                or not isinstance(training_manifest_identity, dict)
                or training_manifest_identity.get("final_checkpoint_verified") is not True
                or not isinstance(training_manifest_identity.get("sha256"), str)
                or not training_manifest_identity.get("sha256")
            ):
                raise SystemExit(
                    f"{group} updated scores must bind a canonical final training checkpoint"
                )
            if model_dtype not in {"fp32", "bf16"} or not isinstance(
                manifest.get("base_model"), dict
            ):
                raise SystemExit(f"{group} {role} lacks a canonical base/dtype identity")
            if common_code is None:
                common_code = code_identity
                common_tokenizer = tokenizer_identity
                common_base_model = base_model_identity
                common_model_dtype = model_dtype
            elif (
                code_identity != common_code
                or tokenizer_identity != common_tokenizer
                or base_model_identity != common_base_model
                or model_dtype != common_model_dtype
            ):
                raise SystemExit(
                    "M3 scores use different code, tokenizer, base model, or dtype"
                )
            if role == "updated":
                updated_adapters.append(adapter_identity)
                updated_training_manifests.append(
                    training_manifest_identity["sha256"]
                )
        score_manifests[group] = group_manifests
        score_rows[group] = tuple(list(read_jsonl(path)) for path in pairs[group])
    if len(set(updated_adapters)) != 2:
        raise SystemExit("A1 and A3 updated adapters must be distinct")
    if len(set(updated_training_manifests)) != 2:
        raise SystemExit("A1 and A3 updated adapters must come from distinct training runs")
    preparation_code = json.dumps(
        [preparation.get("code"), preparation.get("code_revision")],
        sort_keys=True,
        allow_nan=False,
    )
    current_code = json.dumps(
        [fingerprint_code_tree(ROOT), git_revision(ROOT)],
        sort_keys=True,
        allow_nan=False,
    )
    if preparation_code != common_code or current_code != common_code:
        raise SystemExit(
            "M3 preparation, scoring, and assembly must use one code revision"
        )

    metadata = list(read_jsonl(metadata_path))
    assembled = assemble_m3_rows(metadata, score_rows)
    write_jsonl(output, assembled)
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "m3_correction_level_transfer_rows",
        "interpretation": "whole_checkpoint_action_imitation_transfer_not_task_utility",
        "code": fingerprint_code_tree(ROOT),
        "code_revision": git_revision(ROOT),
        "preparation_manifest": {
            "path": str(preparation_path),
            "sha256": sha256_file(preparation_path),
        },
        "score_inputs": {
            group: {
                role: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "manifest_sha256": sha256_file(str(path) + ".manifest.json"),
                }
                for role, path in zip(["base", "updated"], pairs[group])
            }
            for group in ["A1", "A3"]
        },
        "updated_training_manifest_sha256": {
            group: score_manifests[group][1]["training_manifest"]["sha256"]
            for group in ["A1", "A3"]
        },
        "token_contract": "assistant_action_content_tokens_v1",
        "rows": len(assembled),
        "groups": {
            group: sum(row["group"] == group for row in assembled)
            for group in ["A1", "A3"]
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
