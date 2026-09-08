import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from omniopd.evaluation import validate_annotation_pair_manifest
from omniopd.provenance import sha256_file, sha256_json
from scripts.analyze_m1_gradients import _validate_group_annotation_pair
from scripts.validate_experiment_pair import build_annotation_pair_manifest


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest(experiment: str, *, states: int, samples: int) -> dict:
    selected = [f"state-{index}" for index in range(states)]
    behavior_artifact = {
        "kind": "directory",
        "files": 3,
        "bytes": 30,
        "tree_sha256": _sha("behavior-student"),
    }
    return {
        "protocol_version": "omniopd-v1",
        "artifact": "teacher_corrections",
        "experiment": experiment,
        "experiment_config": {"kind": "file", "sha256": _sha(f"config:{experiment}")},
        "distinct_states_M": states,
        "teacher_samples_per_state_N": samples,
        "declared_teacher_budget_B": states * samples,
        "actual_teacher_api_calls": states * samples,
        "state_pool_sha256": _sha("pool"),
        "state_pool_manifest_sha256": _sha("pool-manifest"),
        "behavior_student": {
            "state_source": "student",
            "behavior_model": "student-snapshot",
            "behavior_artifact": behavior_artifact,
            "tokenizer": dict(behavior_artifact),
            "inference_runtime": "vllm-test",
            "behavior_sampling": {
                "thinking_mode": "disabled",
                "temperature": 0.0,
                "reasoning_effort": None,
            },
            "prompt_sha256": _sha("student-prompt"),
            "provider_response_models": ["student-snapshot"],
            "behavior_service_manifest_sha256": _sha(
                "state-pool-service-manifest"
            ),
            "behavior_service_attestation": {
                "attestation_scope": "local_wrapper_child_process",
                "service_role": "student_state_pool",
                "inference_runtime": "vllm-test",
                "served_model": "student-snapshot",
                "max_model_len": 8192,
                "model_artifact": dict(behavior_artifact),
                "tokenizer_artifact": dict(behavior_artifact),
                "launch_command_sha256": _sha("state-pool-launch"),
                "trust_boundary": (
                    "host_local_process_observation_not_cryptographic_remote_attestation"
                ),
            },
        },
        "selection_sha256": _sha(f"selection:{experiment}"),
        "selection_manifest_sha256": _sha(f"selection-manifest:{experiment}"),
        "corrections_sha256": _sha(f"corrections:{experiment}"),
        "teacher_model": "teacher-snapshot",
        "teacher_model_revision": "revision",
        "teacher_url": "https://teacher.test/v1",
        "code": {"tree_sha256": _sha("code"), "files": 5, "bytes": 50},
        "code_revision": "git-revision",
        "teacher_sampling": {"thinking_mode": "disabled", "temperature": 1.0},
        "teacher_max_tokens": 256,
        "teacher_context_window": 8192,
        "teacher_tokenizer": {
            "kind": "directory",
            "files": 2,
            "bytes": 20,
            "tree_sha256": _sha("tokenizer"),
        },
        "teacher_prompt_sha256": _sha("prompt"),
        "provider_response_models": ["teacher-snapshot"],
        "provider_system_fingerprints": ["provider-build"],
        "teacher_context_preflight": {"passed": True},
        "selection_policies": ["uniform_per_game_nested_v1"],
        "selected_state_hashes": selected,
        "selected_counts_by_game": {
            "g1": states // 2,
            "g2": states - states // 2,
        },
        "invalid_calls_count_toward_budget": True,
        "free_retries": 0,
    }


def test_annotation_pair_manifest_binds_both_realized_members():
    with TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        breadth = _manifest("breadth", states=6, samples=1)
        depth = _manifest("depth", states=2, samples=3)
        breadth_path = tmp_path / "breadth.json"
        depth_path = tmp_path / "depth.json"
        breadth_path.write_text(json.dumps(breadth), encoding="utf-8")
        depth_path.write_text(json.dumps(depth), encoding="utf-8")

        pair = build_annotation_pair_manifest(
            [breadth_path, depth_path], [breadth, depth]
        )

        assert pair["artifact"] == "omniopd_annotation_pair"
        assert pair["schema_version"] == 2
        assert pair["pair_contract_sha256"] == sha256_json(pair["pair_contract"])
        assert set(pair["pair_contract"]["members"]) == {"breadth", "depth"}
        assert pair["pair_contract"]["arm_roles"] == {
            "breadth": "breadth",
            "depth": "depth",
        }
        assert pair["pair_contract"]["effect_direction"] == "breadth_minus_depth"
        assert pair["pair_contract"]["members"]["breadth"][
            "corrections_sha256"
        ] == _sha("corrections:breadth")


def test_m1_groups_must_reproduce_the_same_annotation_pair_members():
    with TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        breadth = _manifest("breadth", states=6, samples=1)
        depth = _manifest("depth", states=2, samples=3)
        breadth_path = tmp_path / "breadth.json"
        depth_path = tmp_path / "depth.json"
        breadth_path.write_text(json.dumps(breadth), encoding="utf-8")
        depth_path.write_text(json.dumps(depth), encoding="utf-8")
        pair = build_annotation_pair_manifest(
            [breadth_path, depth_path], [breadth, depth]
        )

        validated = _validate_group_annotation_pair(
            pair,
            pair_manifest_sha256=_sha("pair-file"),
            group_manifests={"A1": breadth, "A3": depth},
            group_manifest_sha256s={
                "A1": sha256_file(breadth_path),
                "A3": sha256_file(depth_path),
            },
        )
        assert validated["group_experiments"] == {
            "A1": "breadth",
            "A3": "depth",
        }

        substituted = dict(depth)
        substituted["corrections_sha256"] = _sha("substituted-corrections")
        try:
            _validate_group_annotation_pair(
                pair,
                pair_manifest_sha256=_sha("pair-file"),
                group_manifests={"A1": breadth, "A3": substituted},
                group_manifest_sha256s={
                    "A1": sha256_file(breadth_path),
                    "A3": sha256_file(depth_path),
                },
            )
        except ValueError as error:
            assert "reproduce both annotation-pair members" in str(error)
        else:
            raise AssertionError("M1 cannot substitute a correction artifact outside the pair")


def test_annotation_pair_rejects_same_pool_bytes_with_different_pool_manifest():
    with TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        breadth = _manifest("breadth", states=6, samples=1)
        depth = _manifest("depth", states=2, samples=3)
        depth["state_pool_manifest_sha256"] = _sha("different-provenance")
        breadth_path = tmp_path / "breadth.json"
        depth_path = tmp_path / "depth.json"
        breadth_path.write_text(json.dumps(breadth), encoding="utf-8")
        depth_path.write_text(json.dumps(depth), encoding="utf-8")

        try:
            build_annotation_pair_manifest(
                [breadth_path, depth_path], [breadth, depth]
            )
        except ValueError as error:
            assert "same state-pool manifest" in str(error)
        else:
            raise AssertionError(
                "state-pool bytes alone cannot replace their provenance manifest"
            )


def test_annotation_pair_rejects_behavior_student_drift_between_arms():
    with TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        breadth = _manifest("breadth", states=6, samples=1)
        depth = _manifest("depth", states=2, samples=3)
        depth["behavior_student"] = {
            **depth["behavior_student"],
            "behavior_model": "different-student",
            "provider_response_models": ["different-student"],
        }
        paths = [tmp_path / "breadth.json", tmp_path / "depth.json"]
        for path, manifest in zip(paths, (breadth, depth)):
            path.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            build_annotation_pair_manifest(paths, [breadth, depth])
        except ValueError as error:
            assert "complete Teacher/state-pool contract" in str(error)
        else:
            raise AssertionError("two annotation arms cannot claim different behavior Students")


def test_annotation_pair_rejects_third_arm_or_missing_shared_teacher_field():
    with TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        breadth = _manifest("breadth", states=6, samples=1)
        depth = _manifest("depth", states=2, samples=3)
        paths = [tmp_path / "breadth.json", tmp_path / "depth.json"]
        for path, manifest in zip(paths, (breadth, depth)):
            path.write_text(json.dumps(manifest), encoding="utf-8")
        pair = build_annotation_pair_manifest(paths, [breadth, depth])

        pair["pair_contract"]["members"]["third"] = {
            **pair["pair_contract"]["members"]["depth"],
            "experiment": "third",
        }
        pair["pair_contract_sha256"] = sha256_json(pair["pair_contract"])
        try:
            validate_annotation_pair_manifest(pair)
        except ValueError as error:
            assert "exactly two arms" in str(error)
        else:
            raise AssertionError("a third annotation arm cannot be hidden in the pair")

        pair = build_annotation_pair_manifest(paths, [breadth, depth])
        del pair["pair_contract"]["shared_contract"]["teacher_tokenizer"]
        pair["pair_contract_sha256"] = sha256_json(pair["pair_contract"])
        try:
            validate_annotation_pair_manifest(pair)
        except ValueError as error:
            assert "incomplete" in str(error)
        else:
            raise AssertionError("the shared Teacher contract cannot omit its tokenizer")
