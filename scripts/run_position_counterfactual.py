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

from omniopd.adapters import AlfworldEnvironment, OpenAIChatPolicy
from omniopd.context import TaskPreservingTruncator
from omniopd.counterfactual import run_paired_counterfactual
from omniopd.io import correction_from_dict, read_jsonl
from omniopd.protocol import GenerationSettings
from omniopd.provenance import (
    fingerprint_code_tree,
    fingerprint_path,
    git_revision,
    sha256_file,
)
from omniopd.validation import audit_correction_records


def _prefix_actions(record) -> list[str]:
    actions = []
    for message in record.state.full_messages[2::2]:
        content = message.get("content", "")
        if message.get("role") != "assistant" or not content.startswith("Action: "):
            raise ValueError("state full history contains a malformed action")
        actions.append(content[len("Action: ") :])
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run symmetric Student/Teacher one-action counterfactual replays"
    )
    parser.add_argument("--corrections", required=True)
    parser.add_argument("--correction-manifest", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--student-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--inference-runtime", required=True)
    parser.add_argument("--student-artifact", action="append", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--reserve-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--rollout-seed", type=int, required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    input_paths = [
        Path(args.corrections),
        Path(args.correction_manifest),
        Path(args.env_config),
        Path(args.tokenizer),
        *(Path(path) for path in args.student_artifact),
    ]
    if any(not path.exists() for path in input_paths):
        raise SystemExit("all counterfactual inputs and model artifacts must exist")
    if output.exists():
        raise SystemExit(f"refusing to overwrite counterfactual output: {output}")
    if (
        args.max_steps <= 0
        or args.rollout_seed < 0
        or args.max_tokens <= 0
        or args.reserve_tokens < args.max_tokens
        or args.max_context_tokens <= args.reserve_tokens
        or not args.inference_runtime.strip()
    ):
        raise SystemExit(
            "steps/tokens must be positive, rollout seed non-negative, and context/reserve "
            "limits must be coherent"
        )

    records = [correction_from_dict(row) for row in read_jsonl(args.corrections)]
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    correction_manifest_path = Path(args.correction_manifest)
    correction_manifest = json.loads(
        correction_manifest_path.read_text(encoding="utf-8")
    )
    if (
        not current_revision
        or correction_manifest.get("artifact") != "teacher_corrections"
        or correction_manifest.get("protocol_version") != "omniopd-v1"
        or correction_manifest.get("corrections_sha256")
        != sha256_file(args.corrections)
        or correction_manifest.get("code") != current_code
        or correction_manifest.get("code_revision") != current_revision
    ):
        raise SystemExit(
            "counterfactual corrections must match a canonical manifest from this revision"
        )
    issues = audit_correction_records(records)
    if issues:
        raise SystemExit(
            f"correction audit failed before counterfactual replay: {issues[:5]}"
        )
    pair_count = sum(len(record.valid_teacher_samples) for record in records)
    if pair_count == 0:
        raise SystemExit("no valid Teacher actions are available for counterfactual replay")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    truncator = TaskPreservingTruncator(
        tokenizer,
        max_context_tokens=args.max_context_tokens,
        reserve_tokens=args.reserve_tokens,
    )
    env_config = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    requests_partial = output / "student_requests.partial.jsonl"
    results_partial = output / "counterfactuals.partial.jsonl"
    policy = OpenAIChatPolicy(
        model=args.student_model,
        base_url=args.student_url,
        api_key="EMPTY",
        thinking_mode="disabled",
        thinking_control="chat_template",
        seed=args.rollout_seed,
        request_ledger_path=requests_partial,
    )
    results = []
    with results_partial.open("x", encoding="utf-8") as handle:
        for record in records:
            prefix = _prefix_actions(record)
            for sample_index, teacher_sample in enumerate(record.teacher_samples):
                if not teacher_sample.valid:
                    continue

                def env_factory(game=record.state.game_id):
                    return AlfworldEnvironment(env_config, game)

                pair = run_paired_counterfactual(
                    env_factory=env_factory,
                    prefix_actions=prefix,
                    state=record.state,
                    student_action=record.student.executed_action,
                    teacher_action=teacher_sample.executed_action,
                    frozen_student=policy,
                    truncator=truncator,
                    settings=GenerationSettings(
                        temperature=0.0,
                        max_tokens=args.max_tokens,
                    ),
                    max_steps=args.max_steps,
                    request_namespace=(
                        f"cf:{record.state.state_hash}:sample:{sample_index}"
                    ),
                )
                if (
                    record.student.executed_action == teacher_sample.executed_action
                    and pair.student_branch != pair.teacher_branch
                ):
                    raise RuntimeError(
                        "identical intervention actions produced asymmetric branch traces"
                    )
                row = {
                    "protocol_version": "omniopd-v1",
                    "state_hash": record.state.state_hash,
                    "game_id": record.state.game_id,
                    "turn_index": record.state.turn_index,
                    "teacher_sample_index": sample_index,
                    "student_action": record.student.executed_action,
                    "teacher_action": teacher_sample.executed_action,
                    "student_branch": pair.student_branch.__dict__,
                    "teacher_branch": pair.teacher_branch.__dict__,
                    "consequence": pair.consequence,
                }
                results.append(row)
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())

    result_path = output / "counterfactuals.jsonl"
    requests_path = output / "student_requests.jsonl"
    if len(results) != pair_count:
        raise RuntimeError("counterfactual output count does not match valid Teacher actions")
    expected_requests = sum(
        row["student_branch"]["continuation_steps"]
        + row["teacher_branch"]["continuation_steps"]
        for row in results
    )
    if len(policy.request_ledger) != expected_requests:
        raise RuntimeError(
            "counterfactual request ledger does not match both continuation branches"
        )
    request_ids = [str(row["request_id"]) for row in policy.request_ledger]
    if len(request_ids) != len(set(request_ids)):
        raise RuntimeError("counterfactual continuation request IDs are not unique")
    provider_response_models = sorted(
        {
            str(row["response_model"])
            for row in policy.request_ledger
            if row.get("response_model") is not None
        }
    )
    if expected_requests and provider_response_models != [args.student_model]:
        raise RuntimeError(
            "counterfactual provider did not return the requested stable Student alias"
        )
    results_partial.rename(result_path)
    requests_partial.rename(requests_path)
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "symmetric_position_counterfactuals",
        "interpretation": "single_action_consequentiality_not_full_training_utility",
        "code": current_code,
        "code_revision": current_revision,
        "corrections_sha256": sha256_file(args.corrections),
        "correction_manifest_sha256": sha256_file(correction_manifest_path),
        "env_config_sha256": sha256_file(args.env_config),
        "student_model": args.student_model,
        "inference_runtime": args.inference_runtime,
        "student_artifacts": [
            fingerprint_path(path) for path in args.student_artifact
        ],
        "tokenizer": fingerprint_path(args.tokenizer),
        "thinking_mode": "disabled",
        "temperature": 0.0,
        "rollout_seed": args.rollout_seed,
        "max_steps": args.max_steps,
        "max_context_tokens": args.max_context_tokens,
        "reserve_tokens": args.reserve_tokens,
        "max_tokens": args.max_tokens,
        "pairs": len(results),
        "continuation_requests": expected_requests,
        "provider_response_models": provider_response_models,
        "provider_system_fingerprints": sorted(
            {
                str(row["system_fingerprint"])
                for row in policy.request_ledger
                if row.get("system_fingerprint") is not None
            }
        ),
        "counterfactuals_sha256": sha256_file(result_path),
        "student_requests_sha256": sha256_file(requests_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
