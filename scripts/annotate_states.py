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

from omniopd.adapters import (
    OpenAIChatPolicy,
    validate_provider_response_model_identity,
)
from omniopd.environment_provenance import content_fingerprint_identity
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
from omniopd.validation import (
    state_pool_behavior_student_contract,
    validate_actual_calls,
    validate_budget,
    validate_selection_manifest_against_rows,
    validate_uncertainty_score_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a frozen state selection with N draws")
    parser.add_argument("--state-pool", required=True)
    parser.add_argument("--state-pool-manifest", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--scores")
    parser.add_argument("--scores-manifest")
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
        configured_selection_seed_raw = experiment.get("selection_seed")
        configured_games = int(experiment["games"])
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
    if configured_selection == "uniform_per_game_nested_v1":
        if isinstance(configured_selection_seed_raw, bool):
            raise SystemExit("experiment.selection_seed must be an integer, not boolean")
        try:
            configured_selection_seed = int(configured_selection_seed_raw)
        except (TypeError, ValueError) as error:
            raise SystemExit("uniform selection requires experiment.selection_seed") from error
        if configured_selection_seed < 0:
            raise SystemExit("experiment.selection_seed must be non-negative")
    else:
        configured_selection_seed = None

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
    try:
        behavior_student = state_pool_behavior_student_contract(state_pool_manifest)
    except ValueError as error:
        raise SystemExit(f"annotation requires the frozen behavior Student: {error}") from error
    turns = [rollout_turn_from_dict(row) for row in read_jsonl(state_pool_path)]
    by_hash = {turn.state.state_hash: turn for turn in turns}
    if not turns or len(by_hash) != len(turns):
        raise SystemExit("state pool must be non-empty and contain unique hashes")
    score_map = None
    if configured_selection == "top_score_per_game":
        if not args.scores or not args.scores_manifest:
            raise SystemExit("top-score annotation requires --scores and --scores-manifest")
        scores_path = Path(args.scores)
        scores_manifest_path = Path(args.scores_manifest)
        if not scores_path.is_file() or not scores_manifest_path.is_file():
            raise SystemExit("top-score files must exist")
        scores_manifest = json.loads(
            scores_manifest_path.read_text(encoding="utf-8")
        )
        expected_scores_manifest = {
            "artifact": "state_uncertainty_scores",
            "protocol_version": "omniopd-v1",
            "code": current_code,
            "code_revision": current_revision,
            "state_pool_sha256": state_pool_sha256,
            "state_pool_manifest_sha256": sha256_file(state_pool_manifest_path),
            "scores_sha256": sha256_file(scores_path),
            "states": len(turns),
            "target_tokenization": (
                "canonical_final_assistant_content_excluding_terminator"
            ),
            "enable_thinking": False,
        }
        score_manifest_mismatches = {
            key: (scores_manifest.get(key), expected)
            for key, expected in expected_scores_manifest.items()
            if scores_manifest.get(key) != expected
        }
        if score_manifest_mismatches:
            raise SystemExit(
                "top-score manifest is not bound to this pool/code: "
                f"{score_manifest_mismatches}"
            )
        try:
            score_model_identity = content_fingerprint_identity(
                scores_manifest.get("model", {})
            )
            score_tokenizer_identity = content_fingerprint_identity(
                scores_manifest.get("tokenizer", {})
            )
            behavior_model_identity = content_fingerprint_identity(
                behavior_student["behavior_artifact"]
            )
            behavior_tokenizer_identity = content_fingerprint_identity(
                behavior_student["tokenizer"]
            )
        except ValueError as error:
            raise SystemExit(
                f"top-score model/tokenizer fingerprint is invalid: {error}"
            ) from error
        if (
            score_model_identity != behavior_model_identity
            or score_tokenizer_identity != behavior_tokenizer_identity
            or not isinstance(scores_manifest.get("device"), str)
            or not scores_manifest["device"].strip()
        ):
            raise SystemExit(
                "top-score artifact must use the frozen behavior Student and tokenizer"
            )
        score_rows = list(read_jsonl(scores_path))
        pool_states = {
            turn.state.state_hash: {
                "game_id": turn.state.game_id,
                "turn_index": turn.state.turn_index,
                "admissible_actions": turn.state.admissible_actions,
            }
            for turn in turns
        }
        try:
            score_map = validate_uncertainty_score_rows(
                score_rows, pool_states=pool_states
            )
        except ValueError as error:
            raise SystemExit(f"top-score rows are invalid: {error}") from error
    elif args.scores or args.scores_manifest:
        raise SystemExit("score files are only valid for top-score annotation")
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
        "behavior_student": behavior_student,
        "policy": configured_selection,
        "states_per_game": configured_states_per_game,
        "distinct_states_M": configured_states,
        "score_file_sha256": sha256_file(args.scores) if args.scores else None,
        "score_manifest_sha256": (
            sha256_file(args.scores_manifest) if args.scores_manifest else None
        ),
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
    pool_state_hashes_by_game: dict[str, list[str]] = {}
    for turn in sorted(turns, key=lambda value: (value.state.game_id, value.state.turn_index)):
        pool_state_hashes_by_game.setdefault(turn.state.game_id, []).append(
            turn.state.state_hash
        )
    try:
        validate_selection_manifest_against_rows(
            selection_manifest,
            selections,
            pool_state_hashes_by_game=pool_state_hashes_by_game,
            score_by_state=score_map,
            expected_experiment=str(experiment["experiment"]),
            expected_policy=configured_selection,
            expected_seed=configured_selection_seed,
            expected_games=configured_games,
            expected_states_per_game=configured_states_per_game,
            expected_states=configured_states,
        )
    except ValueError as error:
        raise SystemExit(f"selection design validation failed: {error}") from error
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
    provider_response_models = validate_provider_response_model_identity(
        teacher.request_ledger,
        args.teacher_model,
    )
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
        "behavior_student": behavior_student,
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
        "provider_response_models": provider_response_models,
        "provider_revision_evidence": {
            "kind": "operator_supplied_provider_snapshot_label",
            "cryptographically_verified": False,
        },
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
