import json
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from omniopd.io import read_jsonl, write_jsonl
from omniopd.prompts import (
    STUDENT_SYSTEM_PROMPT,
    initial_user_message,
    turn_user_message,
)
from omniopd.provenance import (
    fingerprint_code_tree,
    fingerprint_path,
    git_revision,
    sha256_file,
    sha256_json,
    sha256_text,
)
from omniopd.schema import ActionSample, AgentState, RolloutTurn
from omniopd.selection import admissible_entropy
from omniopd.validation import state_pool_behavior_student_contract
from scripts import annotate_states, build_opd_prompts, select_states
from scripts.validate_experiment_pair import build_annotation_pair_manifest


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        del add_generation_prompt, enable_thinking
        if not tokenize:
            return json.dumps(messages, sort_keys=True)
        return list(range(1, 2 + len(messages)))


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(_path, *, trust_remote_code):
        assert trust_remote_code is True
        return _FakeTokenizer()


class _FakeTeacherPolicy:
    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.request_ledger = []
        self.request_ledger_path = Path(kwargs["request_ledger_path"])
        self.request_ledger_path.touch(exist_ok=False)

    def generate(self, messages, *, temperature, max_tokens, request_id):
        entry = {
            "request_id": request_id,
            "status": "ok",
            "response_model": self.model,
            "system_fingerprint": "test-provider-build",
            "messages_sha256": sha256_json(list(messages)),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self.request_ledger.append(entry)
        with self.request_ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return "Action: look"


def _turn(game: str, turn_index: int) -> RolloutTurn:
    task = f"finish {game}"
    actions = ("look", "inventory")
    messages = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": initial_user_message(task, "observation-0", actions),
        },
    ]
    for previous in range(turn_index):
        messages.append({"role": "assistant", "content": "Action: look"})
        messages.append(
            {
                "role": "user",
                "content": turn_user_message(
                    f"observation-{previous + 1}", actions
                ),
            }
        )
    state = AgentState(
        task=task,
        observation=f"observation-{turn_index}",
        admissible_actions=actions,
        messages=tuple(messages),
        game_id=game,
        task_type="pick_and_place",
        turn_index=turn_index,
        full_messages=tuple(messages),
        truncation={
            "token_count": len(messages),
            "dropped_pairs": 0,
            "kept_pairs": turn_index,
            "budget": 3840,
        },
        state_source="student",
    )
    student = ActionSample("Action: look", "look", True, True, None)
    return RolloutTurn(state, student)


def _run_main(module, arguments):
    previous = sys.argv
    sys.argv = [str(module.__file__), *arguments]
    try:
        module.main()
    finally:
        sys.argv = previous


def test_state_pool_selection_annotation_pair_preserves_behavior_student_identity():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        model = root / "behavior_student"
        model.mkdir()
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        teacher_tokenizer = root / "teacher_tokenizer"
        teacher_tokenizer.mkdir()
        (teacher_tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")

        turns = [_turn(game, index) for game in ("game-1", "game-2") for index in range(3)]
        state_pool = root / "state_pool.jsonl"
        write_jsonl(state_pool, (turn.to_dict() for turn in turns))
        model_fingerprint = fingerprint_path(model)
        behavior_sampling = {
            "thinking_mode": "disabled",
            "temperature": 0.0,
            "reasoning_effort": None,
        }
        service_identity = {
            "attestation_scope": "local_wrapper_child_process",
            "service_role": "student_state_pool",
            "inference_runtime": "vllm:test",
            "served_model": "student-test",
            "max_model_len": 8192,
            "model_artifact": model_fingerprint,
            "tokenizer_artifact": model_fingerprint,
            "launch_command_sha256": sha256_text("state-pool-launch"),
            "trust_boundary": (
                "host_local_process_observation_not_cryptographic_remote_attestation"
            ),
        }
        state_pool_manifest = {
            "protocol_version": "omniopd-v1",
            "artifact": "immutable_state_pool",
            "code": fingerprint_code_tree(select_states.ROOT),
            "code_revision": git_revision(select_states.ROOT),
            "state_source": "student",
            "behavior_model": "student-test",
            "inference_runtime": "vllm:test",
            "behavior_sampling": behavior_sampling,
            "provider_response_models": ["student-test"],
            "prompt_sha256": sha256_text(STUDENT_SYSTEM_PROMPT),
            "behavior_artifacts": [model_fingerprint],
            "tokenizer": model_fingerprint,
            "behavior_service_manifest_sha256": sha256_text(
                "state-pool-service-manifest"
            ),
            "behavior_service_attestation": service_identity,
            "games_G": 2,
            "states": len(turns),
            "seed": 42,
            "environment_rollout": {"master_seed": 314159},
            "max_steps": 50,
            "max_context_tokens": 4096,
            "reserve_tokens": 256,
            "behavior_max_tokens": 256,
            "outputs": {"state_pool_sha256": sha256_file(state_pool)},
        }
        state_pool_manifest_path = root / "state_pool.manifest.json"
        state_pool_manifest_path.write_text(
            json.dumps(state_pool_manifest), encoding="utf-8"
        )
        incomplete_pool_manifest = dict(state_pool_manifest)
        incomplete_pool_manifest.pop("behavior_service_attestation")
        try:
            state_pool_behavior_student_contract(incomplete_pool_manifest)
        except ValueError as error:
            assert "behavior Student identity" in str(error)
        else:
            raise AssertionError("a Student pool without service attestation must fail")
        expected_behavior = state_pool_behavior_student_contract(
            state_pool_manifest
        )

        profile = root / "teacher_profile.yaml"
        profile.write_text(
            yaml.safe_dump(
                {
                    "profile": "teacher-test",
                    "thinking_mode": "disabled",
                    "temperature": 1.0,
                    "reasoning_effort": None,
                    "max_tokens": 16,
                }
            ),
            encoding="utf-8",
        )
        common_config = {
            "protocol_version": "omniopd-v1",
            "teacher_budget_definition": "annotation_api_attempts",
            "games": 2,
            "teacher_budget_B": 6,
            "teacher_sampling_profile": "teacher-test",
            "teacher_max_tokens": 16,
            "selection": "uniform_per_game_nested_v1",
            "selection_seed": 42,
            "state_pool": {
                "state_source": "student",
                "games_G": 2,
                "seed": 42,
                "environment_seed": 314159,
                "max_steps": 50,
                "max_context_tokens": 4096,
                "reserve_tokens": 256,
                "behavior_max_tokens": 256,
                "behavior_sampling": behavior_sampling,
            },
        }
        arms = {
            "breadth": {"states_per_game": 3, "distinct_states_M": 6, "teacher_samples_per_state_N": 1},
            "depth": {"states_per_game": 1, "distinct_states_M": 2, "teacher_samples_per_state_N": 3},
        }

        original_policy = annotate_states.OpenAIChatPolicy
        original_transformers = sys.modules.get("transformers")
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = _FakeAutoTokenizer
        annotate_states.OpenAIChatPolicy = _FakeTeacherPolicy
        sys.modules["transformers"] = fake_transformers
        annotation_paths = []
        try:
            for experiment, arm in arms.items():
                config_path = root / f"{experiment}.yaml"
                config_path.write_text(
                    yaml.safe_dump(
                        {**common_config, "experiment": experiment, **arm}
                    ),
                    encoding="utf-8",
                )
                selection_path = root / f"{experiment}.selection.jsonl"
                selection_manifest_path = root / f"{experiment}.selection.manifest.json"
                _run_main(
                    select_states,
                    [
                        "--state-pool",
                        str(state_pool),
                        "--state-pool-manifest",
                        str(state_pool_manifest_path),
                        "--experiment-config",
                        str(config_path),
                        "--output",
                        str(selection_path),
                        "--manifest-output",
                        str(selection_manifest_path),
                    ],
                )
                selection_manifest = json.loads(
                    selection_manifest_path.read_text(encoding="utf-8")
                )
                assert selection_manifest["behavior_student"] == expected_behavior

                opd_prompts_path = root / f"{experiment}.opd_prompts.jsonl"
                opd_manifest_path = root / f"{experiment}.opd_prompts.manifest.json"
                build_opd_prompts.build(
                    state_pool_path=state_pool,
                    state_pool_manifest_path=state_pool_manifest_path,
                    selection_path=selection_path,
                    selection_manifest_path=selection_manifest_path,
                    experiment_path=config_path,
                    output=opd_prompts_path,
                    manifest_output=opd_manifest_path,
                )
                opd_rows = list(read_jsonl(opd_prompts_path))
                assert len(opd_rows) == arm["distinct_states_M"]
                assert all(row["data_source"] == "alfworld_fixed_state_token_opd" for row in opd_rows)
                assert all(
                    row["prompt"][0]["content"] == STUDENT_SYSTEM_PROMPT
                    and row["extra_info"]["teacher_prompt"][0]["content"]
                    != STUDENT_SYSTEM_PROMPT
                    for row in opd_rows
                )
                opd_manifest = json.loads(opd_manifest_path.read_text(encoding="utf-8"))
                assert opd_manifest["training_ready"] is False
                assert opd_manifest["prompts_sha256"] == sha256_file(opd_prompts_path)
                tampered_manifest_path = root / f"{experiment}.tampered_selection.json"
                tampered_manifest_path.write_text(
                    json.dumps({**selection_manifest, "selection_sha256": "0" * 64}),
                    encoding="utf-8",
                )
                try:
                    build_opd_prompts.build(
                        state_pool_path=state_pool,
                        state_pool_manifest_path=state_pool_manifest_path,
                        selection_path=selection_path,
                        selection_manifest_path=tampered_manifest_path,
                        experiment_path=config_path,
                        output=root / f"{experiment}.rejected_opd.jsonl",
                        manifest_output=root / f"{experiment}.rejected_opd.manifest.json",
                    )
                except ValueError as error:
                    assert "selection manifest" in str(error)
                else:
                    raise AssertionError("tampered OPD selection manifest must fail")

                annotation_dir = root / f"{experiment}.annotations"
                _run_main(
                    annotate_states,
                    [
                        "--state-pool",
                        str(state_pool),
                        "--state-pool-manifest",
                        str(state_pool_manifest_path),
                        "--selection",
                        str(selection_path),
                        "--selection-manifest",
                        str(selection_manifest_path),
                        "--experiment-config",
                        str(config_path),
                        "--teacher-profile",
                        str(profile),
                        "--output-dir",
                        str(annotation_dir),
                        "--teacher-model",
                        "teacher-test",
                        "--teacher-model-revision",
                        "teacher-revision-test",
                        "--teacher-url",
                        "https://teacher.test/v1",
                        "--teacher-api-key",
                        "test-only-key",
                        "--teacher-tokenizer",
                        str(teacher_tokenizer),
                        "--teacher-context-window",
                        "128",
                    ],
                )
                annotation_manifest_path = annotation_dir / "manifest.json"
                annotation_manifest = json.loads(
                    annotation_manifest_path.read_text(encoding="utf-8")
                )
                assert annotation_manifest["behavior_student"] == expected_behavior
                annotation_paths.append(annotation_manifest_path)

            score_rows = []
            for turn in turns:
                action_log_scores = {
                    "look": 0.0,
                    "inventory": -float(3 - turn.state.turn_index),
                }
                score_rows.append(
                    {
                        "protocol_version": "omniopd-v1",
                        "state_hash": turn.state.state_hash,
                        "game_id": turn.state.game_id,
                        "turn_index": turn.state.turn_index,
                        "score": admissible_entropy(action_log_scores.values()),
                        "action_log_scores": action_log_scores,
                        "score_definition": (
                            "entropy_of_softmax_sequence_action_logprob"
                        ),
                    }
                )
            scores_path = root / "topscore.scores.jsonl"
            write_jsonl(scores_path, score_rows)
            scores_manifest_path = root / "topscore.scores.manifest.json"
            scores_manifest = {
                "protocol_version": "omniopd-v1",
                "artifact": "state_uncertainty_scores",
                "code": state_pool_manifest["code"],
                "code_revision": state_pool_manifest["code_revision"],
                "model": model_fingerprint,
                "tokenizer": model_fingerprint,
                "device": "test-cpu",
                "states": len(score_rows),
                "state_pool_sha256": sha256_file(state_pool),
                "state_pool_manifest_sha256": sha256_file(
                    state_pool_manifest_path
                ),
                "scores_sha256": sha256_file(scores_path),
                "target_tokenization": (
                    "canonical_final_assistant_content_excluding_terminator"
                ),
                "enable_thinking": False,
            }
            scores_manifest_path.write_text(
                json.dumps(scores_manifest), encoding="utf-8"
            )
            top_config = {
                **common_config,
                "experiment": "topscore",
                "teacher_budget_B": 2,
                "selection": "top_score_per_game",
                "states_per_game": 1,
                "distinct_states_M": 2,
                "teacher_samples_per_state_N": 1,
            }
            top_config.pop("selection_seed")
            top_config_path = root / "topscore.yaml"
            top_config_path.write_text(
                yaml.safe_dump(top_config), encoding="utf-8"
            )
            top_selection_path = root / "topscore.selection.jsonl"
            top_selection_manifest_path = root / "topscore.selection.manifest.json"
            _run_main(
                select_states,
                [
                    "--state-pool",
                    str(state_pool),
                    "--state-pool-manifest",
                    str(state_pool_manifest_path),
                    "--experiment-config",
                    str(top_config_path),
                    "--scores",
                    str(scores_path),
                    "--scores-manifest",
                    str(scores_manifest_path),
                    "--output",
                    str(top_selection_path),
                    "--manifest-output",
                    str(top_selection_manifest_path),
                ],
            )
            top_selection = list(read_jsonl(top_selection_path))
            assert [row["turn_index"] for row in top_selection] == [2, 2]
            assert all(row["inclusion_probability"] is None for row in top_selection)
            assert all(row["score"] == score_rows[index * 3 + 2]["score"] for index, row in enumerate(top_selection))

            top_annotation_dir = root / "topscore.annotations"
            _run_main(
                annotate_states,
                [
                    "--state-pool",
                    str(state_pool),
                    "--state-pool-manifest",
                    str(state_pool_manifest_path),
                    "--selection",
                    str(top_selection_path),
                    "--selection-manifest",
                    str(top_selection_manifest_path),
                    "--scores",
                    str(scores_path),
                    "--scores-manifest",
                    str(scores_manifest_path),
                    "--experiment-config",
                    str(top_config_path),
                    "--teacher-profile",
                    str(profile),
                    "--output-dir",
                    str(top_annotation_dir),
                    "--teacher-model",
                    "teacher-test",
                    "--teacher-model-revision",
                    "teacher-revision-test",
                    "--teacher-url",
                    "https://teacher.test/v1",
                    "--teacher-api-key",
                    "test-only-key",
                    "--teacher-tokenizer",
                    str(teacher_tokenizer),
                    "--teacher-context-window",
                    "128",
                ],
            )
            top_annotation_manifest = json.loads(
                (top_annotation_dir / "manifest.json").read_text(encoding="utf-8")
            )
            assert top_annotation_manifest["behavior_student"] == expected_behavior
            assert top_annotation_manifest["actual_teacher_api_calls"] == 2
        finally:
            annotate_states.OpenAIChatPolicy = original_policy
            if original_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = original_transformers

        annotation_manifests = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in annotation_paths
        ]
        pair = build_annotation_pair_manifest(
            annotation_paths, annotation_manifests
        )
        assert pair["pair_contract"]["shared_contract"][
            "behavior_student"
        ] == expected_behavior
