from omniopd.evaluation import aggregate_evaluation_results, validate_paired_aggregates


def _result(seed, outcomes):
    games = list(outcomes)
    return {
        "protocol_version": "omniopd-v1",
        "code": {"tree_sha256": "same"},
        "code_revision": "revision",
        "training_seed": seed,
        "rollout_seed": 2026,
        "split": "eval_out_of_distribution",
        "games": games,
        "game_list_file_sha256": "games",
        "game_list_manifest_sha256": "games-manifest",
        "game_list_sha256": "canonical-games",
        "env_config_sha256": "env",
        "student_prompt_sha256": "prompt",
        "tokenizer": {"path": f"/seed-{seed}/tokenizer", "tree_sha256": "tok"},
        "model_artifacts": [{"tree_sha256": f"checkpoint-{seed}"}],
        "training_manifest_sha256": f"training-manifest-{seed}",
        "training_protocol": {
            "training_contract": {"total_optimizer_steps": 102},
            "base_model": {"tree_sha256": "base"},
        },
        "inference_runtime": "vllm:test",
        "temperature": 0.0,
        "thinking_mode": "disabled",
        "thinking_control": "chat_template",
        "max_steps": 50,
        "max_context_tokens": 4096,
        "reserve_tokens": 256,
        "max_tokens": 256,
        "provider_response_models": ["served-student"],
        "provider_system_fingerprints": [],
        "request_count": len(outcomes),
        "request_ledger_sha256": f"ledger-{seed}",
        "per_game": outcomes,
        "success_rate": sum(outcomes.values()) / len(outcomes),
    }


def test_evaluation_aggregation_emits_bootstrap_schema_and_ignores_paths():
    bundle, contract = aggregate_evaluation_results(
        {1: _result(1, {"g1": 1, "g2": 0}), 2: _result(2, {"g1": 0, "g2": 1})}
    )
    assert bundle == {1: {"g1": 1.0, "g2": 0.0}, 2: {"g1": 0.0, "g2": 1.0}}
    assert contract["tokenizer"] == {"tree_sha256": "tok"}


def test_evaluation_aggregation_rejects_different_games():
    try:
        aggregate_evaluation_results(
            {1: _result(1, {"g1": 1}), 2: _result(2, {"g2": 1})}
        )
    except ValueError as error:
        assert "game list" in str(error)
    else:
        raise AssertionError("unpaired games cannot enter a paired bootstrap")


def test_evaluation_aggregation_rejects_different_training_exposure():
    first = _result(1, {"g1": 1})
    second = _result(2, {"g1": 0})
    second["training_protocol"] = {
        "training_contract": {"total_optimizer_steps": 999},
        "base_model": {"tree_sha256": "base"},
    }
    try:
        aggregate_evaluation_results({1: first, 2: second})
    except ValueError as error:
        assert "different rollout/game/code contracts" in str(error)
    else:
        raise AssertionError("different optimizer exposure cannot enter one seed aggregate")


def _manifest(games=("g1", "g2"), contract=None):
    contract = contract or {
        "split": "ood",
        "temperature": 0.0,
        "code": {"tree_sha256": "code"},
        "code_revision": "revision",
    }
    return {
        "protocol_version": "omniopd-v1",
        "artifact": "hierarchical_bootstrap_input",
        "code_revision": "revision",
        "seeds": [1, 2],
        "games": len(games),
        "game_ids": sorted(games),
        "code": contract.get("code"),
        "evaluation_contract": contract,
    }


def test_paired_aggregates_require_identical_seed_game_and_rollout_contracts():
    treatment = {1: {"g1": 1, "g2": 0}, 2: {"g1": 0, "g2": 1}}
    control = {1: {"g1": 0, "g2": 0}, 2: {"g1": 0, "g2": 1}}
    pairing = validate_paired_aggregates(
        treatment, control, _manifest(), _manifest()
    )
    assert pairing["seeds"] == [1, 2]
    assert pairing["games"] == ["g1", "g2"]

    changed = _manifest(
        contract={
            "split": "ood",
            "temperature": 0.7,
            "code": {"tree_sha256": "code"},
            "code_revision": "revision",
        }
    )
    try:
        validate_paired_aggregates(treatment, control, _manifest(), changed)
    except ValueError as error:
        assert "different evaluation contracts" in str(error)
    else:
        raise AssertionError("rollout-contract drift must invalidate paired inference")
