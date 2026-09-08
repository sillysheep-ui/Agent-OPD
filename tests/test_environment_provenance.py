from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from omniopd.environment_provenance import (
    REQUIRED_RUNTIME_DISTRIBUTIONS,
    capture_runtime_dependencies,
    derive_environment_seed,
    fingerprint_game_artifacts,
    game_artifacts_digest,
    validate_environment_seed_contract,
    verify_game_artifacts,
    verify_runtime_dependencies,
)


def test_runtime_snapshot_records_required_installed_versions_and_is_strict():
    versions = {name: f"1.0-{name}" for name in REQUIRED_RUNTIME_DISTRIBUTIONS}
    snapshot = capture_runtime_dependencies(version_getter=versions.__getitem__)
    assert snapshot["distributions"] == dict(sorted(versions.items()))
    assert verify_runtime_dependencies(snapshot, actual=snapshot) == snapshot

    changed = {
        **snapshot,
        "distributions": {**snapshot["distributions"], "torch": "different"},
    }
    with pytest.raises(RuntimeError, match="differ"):
        verify_runtime_dependencies(snapshot, actual=changed)


def test_runtime_snapshot_fails_closed_when_distribution_metadata_is_missing():
    def getter(name):
        if name == "textworld":
            raise metadata.PackageNotFoundError(name)
        return "1.0"

    with pytest.raises(RuntimeError, match="textworld"):
        capture_runtime_dependencies(version_getter=getter)


def test_game_artifacts_bind_file_and_sibling_trial_metadata():
    # Avoid pytest-only fixtures so the repository's dependency-light
    # scripts/run_tests.py runner exercises this contract too.
    with TemporaryDirectory() as directory:
        trial = Path(directory) / "trial_1"
        trial.mkdir()
        game = trial / "game.tw-pddl"
        metadata_path = trial / "traj_data.json"
        game.write_text("compiled game", encoding="utf-8")
        metadata_path.write_text(
            '{"task_type":"pick_and_place_simple"}', encoding="utf-8"
        )

        frozen = fingerprint_game_artifacts([game])
        assert frozen[0]["artifact"]["sha256"]
        assert frozen[0]["trial_directory"]["tree_sha256"]
        assert verify_game_artifacts(frozen) == frozen
        assert game_artifacts_digest(frozen) == game_artifacts_digest(frozen)

        metadata_path.write_text('{"task_type":"changed"}', encoding="utf-8")
        with pytest.raises(RuntimeError, match="changed"):
            verify_game_artifacts(frozen)


def test_per_game_environment_seed_is_stable_distinct_and_uint32():
    first = derive_environment_seed(2026, "/games/a")
    assert first == derive_environment_seed(2026, "/games/a")
    assert first != derive_environment_seed(2026, "/games/b")
    assert first != derive_environment_seed(2027, "/games/a")
    assert 0 <= first < 2**32


def test_environment_seed_contract_binds_master_seed_and_ordered_games():
    games = ["/games/a", "/games/b"]
    contract = {
        "master_seed": 2026,
        "derivation": "sha256_omniopd_env_seed_v1_uint32",
        "per_game": [
            {"game_id": game, "seed": derive_environment_seed(2026, game)}
            for game in games
        ],
        "adapter_seed_method": "textworld_gym_env.seed_before_first_reset",
        "alfred_demangler_shuffle": False,
    }
    assert validate_environment_seed_contract(
        contract, game_ids=games, expected_master_seed=2026
    ) == {row["game_id"]: row["seed"] for row in contract["per_game"]}

    changed = {**contract, "master_seed": 2027}
    with pytest.raises(ValueError, match="master seed"):
        validate_environment_seed_contract(
            changed, game_ids=games, expected_master_seed=2026
        )
    with pytest.raises(ValueError, match="ordered game list"):
        validate_environment_seed_contract(
            contract, game_ids=list(reversed(games)), expected_master_seed=2026
        )
