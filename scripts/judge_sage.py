#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import OpenAIChatPolicy
from omniopd.analysis.sage import (
    SAGE_JUDGE_SYSTEM_PROMPT,
    build_sage_judge_messages,
    parse_sage_label,
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
from omniopd.validation import audit_correction_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blindly judge SAGE intervention necessity with exact call accounting"
    )
    parser.add_argument("--corrections", required=True)
    parser.add_argument("--correction-manifest", required=True)
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

    corrections_path = Path(args.corrections)
    correction_manifest_path = Path(args.correction_manifest)
    profile_path = Path(args.judge_profile)
    tokenizer_path = Path(args.judge_tokenizer)
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"refusing to reuse SAGE output directory: {output}")
    if (
        not corrections_path.is_file()
        or not correction_manifest_path.is_file()
        or not tokenizer_path.exists()
    ):
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

    records = [correction_from_dict(row) for row in read_jsonl(corrections_path)]
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    correction_manifest = json.loads(
        correction_manifest_path.read_text(encoding="utf-8")
    )
    if (
        not current_revision
        or correction_manifest.get("artifact") != "teacher_corrections"
        or correction_manifest.get("protocol_version") != "omniopd-v1"
        or correction_manifest.get("corrections_sha256")
        != sha256_file(corrections_path)
        or correction_manifest.get("code") != current_code
        or correction_manifest.get("code_revision") != current_revision
        or not isinstance(correction_manifest.get("state_pool_sha256"), str)
        or not correction_manifest.get("state_pool_sha256")
    ):
        raise SystemExit(
            "SAGE corrections must match a canonical manifest from this code revision"
        )
    issues = audit_correction_records(records)
    if issues:
        raise SystemExit(f"SAGE correction protocol audit failed: {issues[:5]}")
    state_hashes = [record.state.state_hash for record in records]
    if not records or len(state_hashes) != len(set(state_hashes)):
        raise SystemExit("SAGE corrections must contain unique non-empty states")
    try:
        declared_states = int(correction_manifest["distinct_states_M"])
        declared_calls = int(correction_manifest["actual_teacher_api_calls"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"SAGE correction manifest is incomplete: {error}") from error
    if declared_states != len(records) or declared_calls != sum(
        record.teacher_calls for record in records
    ):
        raise SystemExit(
            "SAGE correction rows disagree with the manifest's realized states or calls"
        )
    executable = [record for record in records if record.student.valid]
    if len(executable) != args.judge_budget:
        raise SystemExit(
            "--judge-budget must equal the number of executable states; "
            "technical-invalid states are labeled Strong without an API call"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    query_lengths = {}
    for record in executable:
        messages = build_sage_judge_messages(
            record.state, record.student.executed_action
        )
        query_lengths[record.state.state_hash] = len(
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
        for record in records:
            messages = build_sage_judge_messages(
                record.state, record.student.executed_action
            )
            request_id = f"sage:{record.state.state_hash}"
            raw = None
            label = None
            failure_reason = None
            if record.student.valid:
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
                "state_hash": record.state.state_hash,
                "game_id": record.state.game_id,
                "turn_index": record.state.turn_index,
                "student_action": record.student.executed_action,
                "student_valid": record.student.valid,
                "teacher_samples": [sample.to_dict() for sample in record.teacher_samples],
                "inclusion_probability": record.inclusion_probability,
                "judge_label": label,
                "judge_raw": raw,
                "judge_failure_reason": failure_reason,
                "judge_query_sha256": sha256_json(messages),
                "judge_request_id": request_id if record.student.valid else None,
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
        "corrections_sha256": sha256_file(corrections_path),
        "correction_manifest_sha256": sha256_file(correction_manifest_path),
        "state_pool_sha256": correction_manifest.get("state_pool_sha256"),
        "selection_manifest_sha256": correction_manifest.get(
            "selection_manifest_sha256"
        ),
        "selection_policies": correction_manifest.get("selection_policies"),
        "judge_profile": fingerprint_path(profile_path),
        "judge_model": args.judge_model,
        "judge_model_revision": args.judge_model_revision,
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
        "states": len(rows),
        "technical_states_deterministic_strong": len(rows) - len(executable),
        "declared_judge_budget": args.judge_budget,
        "actual_judge_api_calls": len(policy.request_ledger),
        "free_retries": 0,
        "valid_judge_labels": sum(row["judge_label"] is not None for row in rows),
        "missing_judge_labels": sum(
            row["student_valid"] and row["judge_label"] is None for row in rows
        ),
        "provider_response_models": sorted(
            {
                str(row["response_model"])
                for row in policy.request_ledger
                if row.get("response_model") is not None
            }
        ),
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
