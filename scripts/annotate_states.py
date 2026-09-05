#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import OpenAIChatPolicy
from omniopd.io import read_jsonl, rollout_turn_from_dict
from omniopd.prompts import TEACHER_SYSTEM_PROMPT, replace_system
from omniopd.protocol import GenerationSettings, TeacherBudget, query_teacher
from omniopd.provenance import (
    fingerprint_code_tree,
    fingerprint_path,
    git_revision,
    sha256_file,
)
from omniopd.sampling import SamplingProtocol
from omniopd.tokenization import apply_chat_template_ids
from omniopd.validation import validate_actual_calls, validate_budget


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a frozen state selection with N draws")
    parser.add_argument("--state-pool", required=True)
    parser.add_argument("--state-pool-manifest", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--teacher-profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-model-revision", required=True)
    parser.add_argument("--teacher-url", default="https://api.deepseek.com")
    parser.add_argument("--teacher-api-key")
    parser.add_argument("--teacher-tokenizer", required=True)
    parser.add_argument("--teacher-context-window", type=int, required=True)
    args = parser.parse_args()

    experiment_path = Path(args.experiment_config)
    teacher_profile_path = Path(args.teacher_profile)
    selection_manifest_path = Path(args.selection_manifest)
    state_pool_path = Path(args.state_pool)
    state_pool_manifest_path = Path(args.state_pool_manifest)
    required_inputs = [
        experiment_path,
        teacher_profile_path,
        selection_manifest_path,
        state_pool_path,
        state_pool_manifest_path,
        Path(args.selection),
    ]
    if any(not path.is_file() for path in required_inputs):
        raise SystemExit("annotation config, state pool, selection, and manifests must exist")
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    teacher_profile = yaml.safe_load(
        teacher_profile_path.read_text(encoding="utf-8")
    )
    try:
        configured_states = int(experiment["distinct_states_M"])
        samples_per_state = int(experiment["teacher_samples_per_state_N"])
        teacher_budget = int(experiment["teacher_budget_B"])
        configured_selection = str(experiment["selection"])
        configured_states_per_game = int(experiment["states_per_game"])
        configured_profile = str(experiment["teacher_sampling_profile"])
        max_tokens = int(experiment["teacher_max_tokens"])
        profile_name = str(teacher_profile["profile"])
        thinking_mode = str(teacher_profile["thinking_mode"])
        temperature = teacher_profile.get("temperature")
        reasoning_effort = teacher_profile.get("reasoning_effort")
        profile_max_tokens = int(teacher_profile["max_tokens"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"experiment or Teacher profile is incomplete: {error}") from error
    if configured_profile != profile_name or max_tokens != profile_max_tokens:
        raise SystemExit("experiment and Teacher sampling profile disagree")
    if thinking_mode not in {"enabled", "disabled"}:
        raise SystemExit("Teacher profile must explicitly enable or disable thinking")

    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"refusing to reuse output directory: {output}")
    if not args.teacher_model_revision.strip():
        raise SystemExit("--teacher-model-revision must be a non-empty provider snapshot label")
    teacher_tokenizer_path = Path(args.teacher_tokenizer)
    if not teacher_tokenizer_path.exists():
        raise SystemExit("--teacher-tokenizer must be a local immutable path")
    if max_tokens <= 0 or args.teacher_context_window <= max_tokens:
        raise SystemExit("Teacher context window must exceed the positive generation limit")
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    state_pool_sha256 = sha256_file(state_pool_path)
    state_pool_manifest = json.loads(
        state_pool_manifest_path.read_text(encoding="utf-8")
    )
    if (
        not current_revision
        or state_pool_manifest.get("artifact") != "immutable_state_pool"
        or state_pool_manifest.get("protocol_version") != "omniopd-v1"
        or state_pool_manifest.get("outputs", {}).get("state_pool_sha256")
        != state_pool_sha256
        or state_pool_manifest.get("code") != current_code
        or state_pool_manifest.get("code_revision") != current_revision
    ):
        raise SystemExit(
            "annotation state pool must match a canonical manifest from this revision"
        )
    turns = [rollout_turn_from_dict(row) for row in read_jsonl(state_pool_path)]
    by_hash = {turn.state.state_hash: turn for turn in turns}
    if not turns or len(by_hash) != len(turns):
        raise SystemExit("state pool must be non-empty and contain unique hashes")
    selections = list(read_jsonl(args.selection))
    selection_policies = sorted({str(row.get("selection_policy")) for row in selections})
    selection_hashes = [str(row["state_hash"]) for row in selections]
    if not selections or len(selection_hashes) != len(set(selection_hashes)):
        raise SystemExit("selection must be non-empty and contain unique state hashes")
    selection_manifest = json.loads(
        selection_manifest_path.read_text(encoding="utf-8")
    )
    expected_selection_manifest = {
        "artifact": "selection_manifest",
        "protocol_version": "omniopd-v1",
        "code": current_code,
        "code_revision": current_revision,
        "experiment_config_sha256": sha256_file(experiment_path),
        "selection_sha256": sha256_file(args.selection),
        "state_pool_sha256": state_pool_sha256,
        "state_pool_manifest_sha256": sha256_file(state_pool_manifest_path),
        "policy": configured_selection,
        "states_per_game": configured_states_per_game,
        "distinct_states_M": configured_states,
    }
    mismatches = {
        key: (selection_manifest.get(key), expected)
        for key, expected in expected_selection_manifest.items()
        if selection_manifest.get(key) != expected
    }
    if mismatches:
        raise SystemExit(
            f"selection manifest is not bound to this config/pool/selection: {mismatches}"
        )
    if len(selections) != configured_states or selection_policies != [configured_selection]:
        raise SystemExit("selection rows do not realize the configured M/policy")
    unknown = sorted(set(selection_hashes) - set(by_hash))
    if unknown:
        raise SystemExit(f"selection contains states outside the frozen pool: {unknown[:5]}")
    for row in selections:
        state = by_hash[str(row["state_hash"])].state
        if str(row.get("game_id")) != state.game_id or int(row.get("turn_index")) != state.turn_index:
            raise SystemExit(f"selection metadata mismatch for state {state.state_hash}")

    validate_budget(
        states=len(selections),
        samples_per_state=samples_per_state,
        declared_budget=teacher_budget,
    )
    sampling = SamplingProtocol(
        thinking_mode,
        temperature,
        reasoning_effort,
    )
    sampling.validate(samples_per_state=samples_per_state)
    from transformers import AutoTokenizer

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_tokenizer_path, trust_remote_code=True
    )
    prompt_lengths = {}
    for selection in selections:
        turn = by_hash[str(selection["state_hash"])]
        teacher_messages = replace_system(turn.state.messages, TEACHER_SYSTEM_PROMPT)
        length = len(
            apply_chat_template_ids(
                teacher_tokenizer,
                teacher_messages,
                add_generation_prompt=True,
                enable_thinking=thinking_mode == "enabled",
            )
        )
        prompt_lengths[turn.state.state_hash] = length
    overlength = [
        state_hash
        for state_hash, length in prompt_lengths.items()
        if length + max_tokens > args.teacher_context_window
    ]
    if overlength:
        raise SystemExit(
            "Teacher context preflight failed before any budget was spent; "
            f"overlength states={overlength[:5]}"
        )
    key = args.teacher_api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("provide --teacher-api-key or DEEPSEEK_API_KEY")
    output.mkdir(parents=True, exist_ok=True)
    corrections_partial = output / "corrections.partial.jsonl"
    requests_partial = output / "teacher_requests.partial.jsonl"
    teacher = OpenAIChatPolicy(
        model=args.teacher_model,
        base_url=args.teacher_url,
        api_key=key,
        thinking_mode=sampling.policy_thinking_mode,
        reasoning_effort=sampling.reasoning_effort,
        request_ledger_path=requests_partial,
    )
    budget = TeacherBudget(teacher_budget)
    corrections = []
    with corrections_partial.open("x", encoding="utf-8") as handle:
        for selection in selections:
            turn = by_hash[str(selection["state_hash"])]
            record = query_teacher(
                turn.state,
                turn.student,
                teacher,
                budget,
                samples_per_state=samples_per_state,
                settings=GenerationSettings(
                    temperature=sampling.temperature, max_tokens=max_tokens
                ),
                selection_policy=str(selection["selection_policy"]),
                inclusion_probability=selection.get("inclusion_probability"),
            )
            record.metadata["selection_score"] = selection.get("score")
            corrections.append(record)
            handle.write(
                json.dumps(
                    record.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    validate_actual_calls(corrections, teacher_budget)

    corrections_path = output / "corrections.jsonl"
    requests_path = output / "teacher_requests.jsonl"
    if len(teacher.request_ledger) != teacher_budget:
        raise RuntimeError("Teacher request ledger length does not equal declared budget B")
    ledger_by_request = {
        str(row["request_id"]): row for row in teacher.request_ledger
    }
    if len(ledger_by_request) != len(teacher.request_ledger):
        raise RuntimeError("Teacher request IDs are not unique")
    for record in corrections:
        expected_hash = record.metadata["teacher_query_sha256"]
        for request_id in record.metadata["teacher_request_ids"]:
            if ledger_by_request[request_id]["messages_sha256"] != expected_hash:
                raise RuntimeError(
                    f"Teacher request ledger context mismatch for {request_id}"
                )
    corrections_partial.rename(corrections_path)
    requests_partial.rename(requests_path)
    valid_samples = sum(len(record.valid_teacher_samples) for record in corrections)
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "teacher_corrections",
        "code": current_code,
        "code_revision": current_revision,
        "experiment": experiment["experiment"],
        "experiment_config": fingerprint_path(experiment_path),
        "teacher_profile": fingerprint_path(teacher_profile_path),
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "teacher_model": args.teacher_model,
        "teacher_model_revision": args.teacher_model_revision,
        "teacher_url": args.teacher_url,
        "teacher_sampling": sampling.to_dict(),
        "teacher_max_tokens": max_tokens,
        "teacher_context_window": args.teacher_context_window,
        "teacher_tokenizer": fingerprint_path(teacher_tokenizer_path),
        "teacher_context_preflight": {
            "passed": True,
            "minimum_prompt_tokens": min(prompt_lengths.values()),
            "maximum_prompt_tokens": max(prompt_lengths.values()),
            "all_prompt_plus_generation_within_window": True,
        },
        "provider_response_models": sorted(
            {
                str(row["response_model"])
                for row in teacher.request_ledger
                if row.get("response_model") is not None
            }
        ),
        "provider_system_fingerprints": sorted(
            {
                str(row["system_fingerprint"])
                for row in teacher.request_ledger
                if row.get("system_fingerprint") is not None
            }
        ),
        "teacher_prompt_sha256": hashlib.sha256(
            TEACHER_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "distinct_states_M": len(selections),
        "teacher_samples_per_state_N": samples_per_state,
        "declared_teacher_budget_B": teacher_budget,
        "actual_teacher_api_calls": budget.used_calls,
        "valid_teacher_samples": valid_samples,
        "invalid_teacher_samples": budget.used_calls - valid_samples,
        "invalid_calls_count_toward_budget": True,
        "free_retries": 0,
        "selection_policies": selection_policies,
        "selected_state_hashes": sorted(selection_hashes),
        "selected_counts_by_game": dict(
            sorted(Counter(str(row["game_id"]) for row in selections).items())
        ),
        "state_pool_sha256": state_pool_sha256,
        "state_pool_manifest_sha256": sha256_file(state_pool_manifest_path),
        "selection_sha256": sha256_file(args.selection),
        "corrections_sha256": sha256_file(corrections_path),
        "teacher_requests_sha256": sha256_file(requests_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
