#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.adapters import (
    AlfworldEnvironment,
    OpenAIChatPolicy,
    validate_provider_response_model_identity,
)
from omniopd.context import TaskPreservingTruncator
from omniopd.counterfactual import (
    run_paired_counterfactual,
    summarize_position_counterfactuals,
)
from omniopd.environment_provenance import (
    capture_runtime_dependencies,
    content_fingerprint_identity,
    game_artifacts_digest,
    validate_environment_seed_contract,
    verify_game_artifacts,
    verify_runtime_dependencies,
)
from omniopd.evaluation import validate_position_service_attestation_manifest
from omniopd.io import correction_from_dict, read_jsonl, rollout_turn_from_dict
from omniopd.prompts import STUDENT_SYSTEM_PROMPT
from omniopd.protocol import GenerationSettings
from omniopd.provenance import (
    fingerprint_code_tree,
    fingerprint_path,
    git_revision,
    sha256_file,
)
from omniopd.validation import (
    audit_correction_records,
    validate_correction_manifest_against_records,
)


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
    parser.add_argument("--state-pool", required=True)
    parser.add_argument("--state-pool-manifest", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--student-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--student-service-manifest", required=True)
    parser.add_argument("--inference-runtime", required=True)
    parser.add_argument("--student-artifact", action="append", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--reserve-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--rollout-seed", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=50_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    output = Path(args.output_dir)
    input_paths = [
        Path(args.corrections),
        Path(args.correction_manifest),
        Path(args.state_pool),
        Path(args.state_pool_manifest),
        Path(args.env_config),
        Path(args.student_service_manifest),
        Path(args.tokenizer),
        *(Path(path) for path in args.student_artifact),
    ]
    if any(not path.exists() for path in input_paths):
        raise SystemExit("all counterfactual inputs and model artifacts must exist")
    if len(args.student_artifact) != 1:
        raise SystemExit("Position requires exactly one complete frozen Student artifact")
    if output.exists():
        raise SystemExit(f"refusing to overwrite counterfactual output: {output}")
    if (
        args.max_steps <= 0
        or args.rollout_seed < 0
        or args.bootstrap_seed < 0
        or args.bootstrap_replicates <= 0
        or not 0.0 < args.confidence < 1.0
        or args.max_tokens <= 0
        or args.reserve_tokens < args.max_tokens
        or args.max_context_tokens <= args.reserve_tokens
        or not args.inference_runtime.strip()
    ):
        raise SystemExit(
            "steps/tokens must be positive, rollout seed non-negative, and context/reserve "
            "limits must be coherent"
        )

    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if not current_revision:
        raise SystemExit("Position requires an immutable Git revision")
    try:
        runtime_dependencies = capture_runtime_dependencies()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    records = [correction_from_dict(row) for row in read_jsonl(args.corrections)]
    if not records:
        raise SystemExit("counterfactual corrections must be non-empty")
    record_hashes = [record.state.state_hash for record in records]
    if len(record_hashes) != len(set(record_hashes)):
        raise SystemExit("counterfactual corrections contain duplicate states")

    correction_manifest_path = Path(args.correction_manifest)
    correction_manifest = json.loads(
        correction_manifest_path.read_text(encoding="utf-8")
    )
    state_pool_path = Path(args.state_pool)
    state_pool_manifest_path = Path(args.state_pool_manifest)
    state_pool_manifest = json.loads(
        state_pool_manifest_path.read_text(encoding="utf-8")
    )
    state_pool_sha256 = sha256_file(state_pool_path)
    state_pool_manifest_sha256 = sha256_file(state_pool_manifest_path)
    if (
        correction_manifest.get("artifact") != "teacher_corrections"
        or correction_manifest.get("protocol_version") != "omniopd-v1"
        or correction_manifest.get("corrections_sha256")
        != sha256_file(args.corrections)
        or correction_manifest.get("code") != current_code
        or correction_manifest.get("code_revision") != current_revision
        or correction_manifest.get("state_pool_sha256") != state_pool_sha256
        or correction_manifest.get("state_pool_manifest_sha256")
        != state_pool_manifest_sha256
        or sorted(correction_manifest.get("selected_state_hashes", []))
        != sorted(record_hashes)
    ):
        raise SystemExit(
            "counterfactual corrections must match this exact state pool and canonical revision"
        )
    if (
        state_pool_manifest.get("artifact") != "immutable_state_pool"
        or state_pool_manifest.get("protocol_version") != "omniopd-v1"
        or state_pool_manifest.get("code") != current_code
        or state_pool_manifest.get("code_revision") != current_revision
        or state_pool_manifest.get("outputs", {}).get("state_pool_sha256")
        != state_pool_sha256
        or state_pool_manifest.get("state_source") != "student"
    ):
        raise SystemExit(
            "Position requires the canonical Student behavior state pool from this revision"
        )
    try:
        verify_runtime_dependencies(
            state_pool_manifest.get("runtime_dependencies", {}),
            actual=runtime_dependencies,
        )
        frozen_game_artifacts = state_pool_manifest["game_artifacts"]
        verified_game_artifacts = verify_game_artifacts(frozen_game_artifacts)
    except (KeyError, RuntimeError, ValueError, FileNotFoundError) as error:
        raise SystemExit(f"state-pool environment provenance failed: {error}") from error
    if (
        state_pool_manifest.get("game_artifacts_sha256")
        != game_artifacts_digest(verified_game_artifacts)
    ):
        raise SystemExit("state-pool game-artifact digest is missing or inconsistent")

    turns = [rollout_turn_from_dict(row) for row in read_jsonl(state_pool_path)]
    pool_by_hash = {turn.state.state_hash: turn for turn in turns}
    if not turns or len(pool_by_hash) != len(turns):
        raise SystemExit("state pool must be non-empty and contain unique state hashes")
    population_states_by_game = dict(
        sorted(Counter(turn.state.game_id for turn in turns).items())
    )
    if (
        len(population_states_by_game) != int(state_pool_manifest.get("games_G", -1))
        or len(turns) != int(state_pool_manifest.get("states", -1))
        or set(population_states_by_game)
        != {str(row["game_id"]) for row in verified_game_artifacts}
    ):
        raise SystemExit("state-pool rows, game artifacts, and manifest counts disagree")
    try:
        validate_correction_manifest_against_records(correction_manifest, records)
    except ValueError as error:
        raise SystemExit(
            f"Position correction manifest contradicts its records: {error}"
        ) from error
    for record in records:
        source = pool_by_hash.get(record.state.state_hash)
        if (
            source is None
            or source.state.to_dict() != record.state.to_dict()
            or source.student.to_dict() != record.student.to_dict()
        ):
            raise SystemExit(
                f"correction state is not an exact state-pool row: {record.state.state_hash}"
            )
        probability = record.inclusion_probability
        if record.selection_policy == "uniform_per_game_nested_v1":
            selected_in_game = int(
                correction_manifest["selected_counts_by_game"][record.state.game_id]
            )
            expected_probability = (
                selected_in_game / population_states_by_game[record.state.game_id]
            )
            if (
                probability is None
                or not abs(float(probability) - expected_probability) <= 1e-12
            ):
                raise SystemExit(
                    "Position inclusion_probability does not equal realized m_g/T_g "
                    f"for state {record.state.state_hash}"
                )
        elif probability is not None:
            raise SystemExit(
                "Position cannot claim a design probability for a non-uniform selection"
            )

    student_artifacts = [fingerprint_path(path) for path in args.student_artifact]
    tokenizer_fingerprint = fingerprint_path(args.tokenizer)
    try:
        frozen_artifact_identities = sorted(
            repr(content_fingerprint_identity(item))
            for item in state_pool_manifest["behavior_artifacts"]
        )
        current_artifact_identities = sorted(
            repr(content_fingerprint_identity(item)) for item in student_artifacts
        )
        frozen_tokenizer_identity = content_fingerprint_identity(
            state_pool_manifest["tokenizer"]
        )
        current_tokenizer_identity = content_fingerprint_identity(tokenizer_fingerprint)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid behavior Student artifact provenance: {error}") from error
    behavior_sampling = state_pool_manifest.get("behavior_sampling")
    expected_behavior_sampling = {
        "thinking_mode": "disabled",
        "temperature": 0.0,
        "reasoning_effort": None,
    }
    if (
        args.student_model != state_pool_manifest.get("behavior_model")
        or args.inference_runtime != state_pool_manifest.get("inference_runtime")
        or frozen_artifact_identities != current_artifact_identities
        or frozen_tokenizer_identity != current_tokenizer_identity
        or args.max_context_tokens
        != int(state_pool_manifest.get("max_context_tokens", -1))
        or args.reserve_tokens != int(state_pool_manifest.get("reserve_tokens", -1))
        or args.max_tokens != int(state_pool_manifest.get("behavior_max_tokens", -1))
        or args.max_steps != int(state_pool_manifest.get("max_steps", -1))
        or sha256_file(args.env_config) != state_pool_manifest.get("env_config_sha256")
        or behavior_sampling != expected_behavior_sampling
        or state_pool_manifest.get("prompt_sha256")
        != hashlib.sha256(STUDENT_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    ):
        raise SystemExit(
            "Position Student/tokenizer/context/environment/horizon does not exactly match "
            "the behavior Student state-pool contract"
        )

    student_service_path = Path(args.student_service_manifest)
    try:
        student_service_manifest = json.loads(
            student_service_path.read_text(encoding="utf-8")
        )
        student_service_identity = validate_position_service_attestation_manifest(
            student_service_manifest,
            requested_model=args.student_model,
            base_url=args.student_url,
            inference_runtime=args.inference_runtime,
            model_fingerprint=student_artifacts[0],
            tokenizer_fingerprint=tokenizer_fingerprint,
            wrapper_fingerprint=fingerprint_path(
                ROOT / "scripts/run_vllm_position.sh"
            ),
            expected_parent_pid=os.getppid(),
            require_live_process=True,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid Position Student-service binding: {error}") from error
    if int(student_service_identity["max_model_len"]) < args.max_context_tokens:
        raise SystemExit("attested Position service context is smaller than replay context")

    try:
        environment_rollout = state_pool_manifest["environment_rollout"]
        environment_master_seed = int(environment_rollout["master_seed"])
        environment_seeds = validate_environment_seed_contract(
            environment_rollout,
            game_ids=[
                str(row["game_id"])
                for row in state_pool_manifest["game_artifacts"]
            ],
            expected_master_seed=environment_master_seed,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"state-pool environment seed contract failed: {error}") from error
    if set(environment_seeds) != set(population_states_by_game):
        raise SystemExit("state-pool environment seeds do not cover the state population")

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
            valid_teacher_draws = len(record.valid_teacher_samples)
            inclusion_probability = record.inclusion_probability
            state_weight = (
                None
                if inclusion_probability is None
                else 1.0 / inclusion_probability
            )
            for sample_index, teacher_sample in enumerate(record.teacher_samples):
                if not teacher_sample.valid:
                    continue

                def env_factory(
                    game=record.state.game_id,
                    environment_seed=environment_seeds[record.state.game_id],
                ):
                    return AlfworldEnvironment(
                        env_config,
                        game,
                        rollout_seed=environment_seed,
                    )

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
                    "state_source": record.state.state_source,
                    "selection_policy": record.selection_policy,
                    "inclusion_probability": inclusion_probability,
                    "state_weight": state_weight,
                    "teacher_draw_weight_within_state": 1.0 / valid_teacher_draws,
                    "combined_analysis_weight": (
                        None
                        if state_weight is None
                        else state_weight / valid_teacher_draws
                    ),
                    "environment_rollout_seed": environment_seeds[
                        record.state.game_id
                    ],
                    "teacher_sample_index": sample_index,
                    "student_action": record.student.executed_action,
                    "teacher_action": teacher_sample.executed_action,
                    "disagreement": int(
                        record.student.executed_action
                        != teacher_sample.executed_action
                    ),
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
    provider_response_models = validate_provider_response_model_identity(
        policy.request_ledger,
        args.student_model,
        require_success=expected_requests > 0,
    )
    provider_system_fingerprints = sorted(
        {
            str(row["system_fingerprint"])
            for row in policy.request_ledger
            if row.get("system_fingerprint") is not None
        }
    )
    frozen_system_fingerprints = state_pool_manifest.get(
        "provider_system_fingerprints", []
    )
    if (
        expected_requests
        and frozen_system_fingerprints
        and provider_system_fingerprints != frozen_system_fingerprints
    ):
        raise RuntimeError(
            "counterfactual Student provider fingerprint differs from the behavior rollout"
        )
    summary = summarize_position_counterfactuals(
        results,
        population_states_by_game=population_states_by_game,
        expected_state_hashes=record_hashes,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence=args.confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary_partial = output / "summary.partial.json"
    summary_path = output / "summary.json"
    summary_partial.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    results_partial.rename(result_path)
    requests_partial.rename(requests_path)
    summary_partial.rename(summary_path)
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "symmetric_position_counterfactuals",
        "interpretation": "single_action_consequentiality_not_full_training_utility",
        "code": current_code,
        "code_revision": current_revision,
        "corrections_sha256": sha256_file(args.corrections),
        "correction_manifest_sha256": sha256_file(correction_manifest_path),
        "state_pool_sha256": state_pool_sha256,
        "state_pool_manifest_sha256": state_pool_manifest_sha256,
        "env_config_sha256": sha256_file(args.env_config),
        "game_artifacts_sha256": game_artifacts_digest(verified_game_artifacts),
        "runtime_dependencies": runtime_dependencies,
        "student_model": args.student_model,
        "student_url": args.student_url,
        "inference_runtime": args.inference_runtime,
        "student_artifacts": student_artifacts,
        "tokenizer": tokenizer_fingerprint,
        "student_service_manifest_sha256": sha256_file(student_service_path),
        "student_service_attestation": student_service_identity,
        "thinking_mode": "disabled",
        "temperature": 0.0,
        "student_decoding_seed": args.rollout_seed,
        "environment_rollout": environment_rollout,
        "max_steps": args.max_steps,
        "max_context_tokens": args.max_context_tokens,
        "reserve_tokens": args.reserve_tokens,
        "max_tokens": args.max_tokens,
        "pairs": len(results),
        "selected_states": len(records),
        "states_without_valid_teacher_action": sum(
            not record.valid_teacher_samples for record in records
        ),
        "continuation_requests": expected_requests,
        "provider_response_models": provider_response_models,
        "provider_system_fingerprints": provider_system_fingerprints,
        "counterfactuals_sha256": sha256_file(result_path),
        "student_requests_sha256": sha256_file(requests_path),
        "summary_sha256": sha256_file(summary_path),
        "population_identified": summary["population_identified"],
        "population_nonidentification_reasons": summary[
            "nonidentification_reasons"
        ],
        "primary_estimand": (
            "finite_state_population_game_balanced_E_C_given_D1_and_teacher_valid"
        ),
        "secondary_estimand": (
            "finite_state_population_game_balanced_unconditional_single_action_"
            "consequentiality"
        ),
        "bootstrap": {
            "seed": args.bootstrap_seed,
            "replicates": args.bootstrap_replicates,
            "confidence": args.confidence,
            "resampling_unit": "paired_game_cluster",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
