#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import (
    OpenAIChatPolicy,
    validate_provider_response_model_identity,
)
from omniopd.analysis.gradient import teacher_contract_from_manifest
from omniopd.analysis.sage import (
    SAGE_JUDGE_SYSTEM_PROMPT,
    build_sage_judge_messages,
    parse_sage_label,
    state_level_disagreement,
)
from omniopd.io import correction_from_dict, read_jsonl
from omniopd.provenance import (
    fingerprint_code_tree,
    fingerprint_path,
    git_revision,
    sha256_file,
    sha256_json,
    sha256_text,
)
from omniopd.sampling import SamplingProtocol
from omniopd.tokenization import apply_chat_template_ids
from omniopd.validation import (
    audit_correction_records,
    validate_correction_manifest_against_records,
)


def _load_sage_union(
    correction_paths: list[Path],
    manifest_paths: list[Path],
    *,
    current_code: dict[str, Any],
    current_revision: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Join several correction groups and deduplicate the blind-judge states.

    A state receives one intervention label, while every source-group
    membership retains its own single Teacher draw, design probability and D.
    """

    if not correction_paths or len(correction_paths) != len(manifest_paths):
        raise SystemExit(
            "repeat --corrections and --correction-manifest the same number of times"
        )
    union: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    experiments: set[str] = set()
    shared_pool: str | None = None
    shared_teacher_contract: dict[str, Any] | None = None
    for corrections_path, manifest_path in zip(
        correction_paths, manifest_paths, strict=True
    ):
        if not corrections_path.is_file() or not manifest_path.is_file():
            raise SystemExit("every SAGE correction file and manifest must exist")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        experiment = str(manifest.get("experiment", "")).strip()
        pool_sha256 = manifest.get("state_pool_sha256")
        if (
            manifest.get("artifact") != "teacher_corrections"
            or manifest.get("protocol_version") != "omniopd-v1"
            or manifest.get("corrections_sha256") != sha256_file(corrections_path)
            or manifest.get("code") != current_code
            or manifest.get("code_revision") != current_revision
            or not experiment
            or not isinstance(pool_sha256, str)
            or not pool_sha256
        ):
            raise SystemExit(
                f"SAGE source {manifest_path} is not a canonical correction manifest "
                "from this code revision"
            )
        if experiment in experiments:
            raise SystemExit(f"duplicate SAGE experiment membership: {experiment}")
        experiments.add(experiment)
        if shared_pool is None:
            shared_pool = pool_sha256
        elif pool_sha256 != shared_pool:
            raise SystemExit("all SAGE groups must come from the same frozen state pool")
        try:
            teacher_contract = teacher_contract_from_manifest(manifest)
        except ValueError as error:
            raise SystemExit(
                f"SAGE source {manifest_path} has an invalid Teacher contract: {error}"
            ) from error
        if shared_teacher_contract is None:
            shared_teacher_contract = teacher_contract
        elif teacher_contract != shared_teacher_contract:
            raise SystemExit("all SAGE groups must share one frozen Teacher contract")

        records = [correction_from_dict(row) for row in read_jsonl(corrections_path)]
        issues = audit_correction_records(records)
        if issues:
            raise SystemExit(
                f"SAGE correction protocol audit failed for {experiment}: {issues[:5]}"
            )
        try:
            validate_correction_manifest_against_records(manifest, records)
        except ValueError as error:
            raise SystemExit(
                f"SAGE correction manifest contradicts its records for "
                f"{experiment}: {error}"
            ) from error
        state_hashes = [record.state.state_hash for record in records]
        try:
            declared_states = int(manifest["distinct_states_M"])
            declared_calls = int(manifest["actual_teacher_api_calls"])
            samples_per_state = int(manifest["teacher_samples_per_state_N"])
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"SAGE source manifest is incomplete: {error}") from error
        if (
            not records
            or len(state_hashes) != len(set(state_hashes))
            or declared_states != len(records)
            or declared_calls != len(records)
            or samples_per_state != 1
            or any(
                record.teacher_calls != 1 or len(record.teacher_samples) != 1
                for record in records
            )
        ):
            raise SystemExit(
                "each SAGE group requires unique states and exactly one budgeted "
                "Teacher draw per state"
            )

        source_binding = {
            "experiment": experiment,
            "corrections": fingerprint_path(corrections_path),
            "correction_manifest": fingerprint_path(manifest_path),
            "selection_manifest_sha256": manifest.get("selection_manifest_sha256"),
            "selection_policies": manifest.get("selection_policies"),
            "states": len(records),
            "teacher_contract_sha256": sha256_json(teacher_contract),
        }
        sources.append(source_binding)
        for record in records:
            state_hash = record.state.state_hash
            existing = union.get(state_hash)
            if existing is None:
                existing = {
                    "state": record.state,
                    "student": record.student,
                    "memberships": [],
                }
                union[state_hash] = existing
            elif (
                existing["state"].to_dict() != record.state.to_dict()
                or existing["student"].to_dict() != record.student.to_dict()
            ):
                raise SystemExit(
                    f"overlapping state {state_hash} has different state or Student data"
                )
            existing["memberships"].append(
                {
                    "experiment": experiment,
                    "teacher_samples": [
                        sample.to_dict() for sample in record.teacher_samples
                    ],
                    "teacher_valid": bool(record.valid_teacher_samples),
                    "disagreement": state_level_disagreement(
                        record.student.executed_action, record.teacher_samples
                    ),
                    "selection_policy": record.selection_policy,
                    "inclusion_probability": record.inclusion_probability,
                    "corrections_sha256": manifest["corrections_sha256"],
                    "correction_manifest_sha256": sha256_file(manifest_path),
                }
            )
    if shared_pool is None:
        raise SystemExit("SAGE union is empty")
    return [union[key] for key in sorted(union)], sources, shared_pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blindly judge SAGE intervention necessity with exact call accounting"
    )
    parser.add_argument(
        "--corrections",
        required=True,
        action="append",
        help="repeat once per experiment group",
    )
    parser.add_argument(
        "--correction-manifest",
        required=True,
        action="append",
        help="repeat in the same order as --corrections",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-profile", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-model-revision", required=True)
    parser.add_argument("--judge-url", default="https://api.deepseek.com")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--judge-tokenizer", required=True)
    parser.add_argument("--judge-context-window", type=int, required=True)
    parser.add_argument("--judge-budget", type=int, required=True)
    args = parser.parse_args()

    correction_paths = [Path(path) for path in args.corrections]
    correction_manifest_paths = [Path(path) for path in args.correction_manifest]
    profile_path = Path(args.judge_profile)
    tokenizer_path = Path(args.judge_tokenizer)
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"refusing to reuse SAGE output directory: {output}")
    if not tokenizer_path.exists():
        raise SystemExit(
            "SAGE corrections, correction manifest, and local judge tokenizer must exist"
        )
    if not profile_path.is_file() or not args.judge_model_revision.strip():
        raise SystemExit("SAGE judge profile and non-empty model revision are required")
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    try:
        thinking_mode = str(profile["thinking_mode"])
        temperature = profile.get("temperature")
        reasoning_effort = profile.get("reasoning_effort")
        max_tokens = int(profile["max_tokens"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"SAGE judge profile is incomplete: {error}") from error
    sampling = SamplingProtocol(thinking_mode, temperature, reasoning_effort)
    sampling.validate(samples_per_state=1)
    if thinking_mode not in {"enabled", "disabled"}:
        raise SystemExit("SAGE judge must explicitly enable or disable thinking")
    if (
        args.judge_budget < 0
        or max_tokens <= 0
        or args.judge_context_window <= max_tokens
    ):
        raise SystemExit("SAGE budget/context/generation limits are invalid")

    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if not current_revision:
        raise SystemExit("SAGE requires a Git revision")
    union_rows, source_bindings, shared_pool_sha256 = _load_sage_union(
        correction_paths,
        correction_manifest_paths,
        current_code=current_code,
        current_revision=current_revision,
    )
    executable = [row for row in union_rows if row["student"].valid]
    if len(executable) != args.judge_budget:
        raise SystemExit(
            "--judge-budget must equal the number of executable states; "
            "technical-invalid states are labeled Strong without an API call"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    query_lengths = {}
    for item in executable:
        messages = build_sage_judge_messages(
            item["state"], item["student"].executed_action
        )
        query_lengths[item["state"].state_hash] = len(
            apply_chat_template_ids(
                tokenizer,
                messages,
                add_generation_prompt=True,
                enable_thinking=thinking_mode == "enabled",
            )
        )
    overlength = [
        state_hash
        for state_hash, length in query_lengths.items()
        if length + max_tokens > args.judge_context_window
    ]
    if overlength:
        raise SystemExit(
            "SAGE judge context preflight failed before any calls; "
            f"overlength states={overlength[:5]}"
        )
    key = args.judge_api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("provide --judge-api-key or DEEPSEEK_API_KEY")

    output.mkdir(parents=True)
    labels_partial = output / "labels.partial.jsonl"
    requests_partial = output / "judge_requests.partial.jsonl"
    policy = OpenAIChatPolicy(
        model=args.judge_model,
        base_url=args.judge_url,
        api_key=key,
        thinking_mode=sampling.policy_thinking_mode,
        reasoning_effort=sampling.reasoning_effort,
        request_ledger_path=requests_partial,
    )
    rows = []
    with labels_partial.open("x", encoding="utf-8") as handle:
        for item in union_rows:
            messages = build_sage_judge_messages(
                item["state"], item["student"].executed_action
            )
            request_id = f"sage:{item['state'].state_hash}"
            raw = None
            label = None
            failure_reason = None
            if item["student"].valid:
                try:
                    raw = policy.generate(
                        messages,
                        temperature=sampling.temperature,
                        max_tokens=max_tokens,
                        request_id=request_id,
                    )
                    label = parse_sage_label(raw)
                    if label is None:
                        failure_reason = "invalid_judge_label"
                except Exception as error:
                    failure_reason = f"judge_request_error:{type(error).__name__}"
            row = {
                "protocol_version": "omniopd-v1",
                "state_hash": item["state"].state_hash,
                "game_id": item["state"].game_id,
                "turn_index": item["state"].turn_index,
                "student_action": item["student"].executed_action,
                "student_valid": item["student"].valid,
                "memberships": item["memberships"],
                "judge_label": label,
                "judge_raw": raw,
                "judge_failure_reason": failure_reason,
                "judge_query_sha256": sha256_json(messages),
                "judge_request_id": request_id if item["student"].valid else None,
            }
            rows.append(row)
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    if len(policy.request_ledger) != args.judge_budget:
        raise RuntimeError("SAGE request ledger length does not equal the declared budget")
    ledger = {str(row["request_id"]): row for row in policy.request_ledger}
    if len(ledger) != args.judge_budget:
        raise RuntimeError("SAGE request IDs are not unique")
    provider_response_models = validate_provider_response_model_identity(
        policy.request_ledger,
        args.judge_model,
        require_success=False,
    )
    for row in rows:
        if row["student_valid"] and (
            row["judge_request_id"] not in ledger
            or ledger[row["judge_request_id"]]["messages_sha256"]
            != row["judge_query_sha256"]
        ):
            raise RuntimeError("SAGE ledger context does not match its blind judge query")
    labels_path = output / "labels.jsonl"
    requests_path = output / "judge_requests.jsonl"
    labels_partial.rename(labels_path)
    requests_partial.rename(requests_path)
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "sage_blind_intervention_labels",
        "code": current_code,
        "code_revision": current_revision,
        "sage_input_schema": "union_unique_state_memberships_v1",
        "source_groups": source_bindings,
        "experiments": sorted(source["experiment"] for source in source_bindings),
        "state_pool_sha256": shared_pool_sha256,
        "teacher_samples_per_membership_N": 1,
        "disagreement_definition": "single_teacher_draw_action_inequality",
        "judge_profile": fingerprint_path(profile_path),
        "judge_model": args.judge_model,
        "judge_model_revision": args.judge_model_revision,
        "judge_model_revision_evidence": (
            "provider snapshot label supplied by the operator; the remote API does not "
            "cryptographically attest model weights"
        ),
        "judge_url": args.judge_url,
        "judge_tokenizer": fingerprint_path(tokenizer_path),
        "judge_prompt_sha256": sha256_text(SAGE_JUDGE_SYSTEM_PROMPT),
        "judge_sampling": sampling.to_dict(),
        "judge_max_tokens": max_tokens,
        "judge_context_window": args.judge_context_window,
        "context_preflight": {
            "passed": True,
            "minimum_prompt_tokens": min(query_lengths.values()) if query_lengths else None,
            "maximum_prompt_tokens": max(query_lengths.values()) if query_lengths else None,
        },
        "unique_states": len(rows),
        "group_memberships": sum(len(row["memberships"]) for row in rows),
        "technical_states_deterministic_strong": len(rows) - len(executable),
        "declared_judge_budget": args.judge_budget,
        "actual_judge_api_calls": len(policy.request_ledger),
        "free_retries": 0,
        "valid_judge_labels": sum(row["judge_label"] is not None for row in rows),
        "missing_judge_labels": sum(
            row["student_valid"] and row["judge_label"] is None for row in rows
        ),
        "provider_response_models": provider_response_models,
        "provider_system_fingerprints": sorted(
            {
                str(row["system_fingerprint"])
                for row in policy.request_ledger
                if row.get("system_fingerprint") is not None
            }
        ),
        "labels_sha256": sha256_file(labels_path),
        "judge_requests_sha256": sha256_file(requests_path),
        "teacher_information_sent_to_judge": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
