import tempfile
from pathlib import Path

from omniopd.analysis.m3_panel import assemble_m3_rows, build_m3_panel
from omniopd.schema import ActionSample, AgentState, RolloutTurn
from scripts.assemble_m3 import _validate_checkpoint_source_binding
from scripts.score_action_logp import _resolve_training_tokenizer_path


def _turn(game, turn, observation, *, selected=False, student_valid=True):
    del selected
    state = AgentState(
        task="put object",
        observation=observation,
        admissible_actions=("look", "go north"),
        messages=(
            {"role": "system", "content": "old"},
            {"role": "user", "content": f"state {game} {turn}"},
        ),
        game_id=game,
        task_type="pick_and_place_simple",
        turn_index=turn,
    )
    sample = ActionSample(
        raw="Action: look",
        executed_action="look",
        valid=student_valid,
        had_action_marker=True,
        failure_reason=None if student_valid else "invalid",
    )
    return RolloutTurn(state, sample)


def _source(turn):
    return {
        "state_hash": turn.state.state_hash,
        "game_id": turn.state.game_id,
        "task_type": turn.state.task_type,
        "turn_index": turn.state.turn_index,
        "teacher_sample_index": 0,
        "teacher_action": "look",
        "student_action": turn.student.executed_action,
        "state_source": turn.state.state_source,
        "protocol_version": "omniopd-v1",
        "enable_thinking": False,
    }


def test_m3_panel_freezes_cross_game_nonselected_neighbors_and_assembles_deltas():
    a1 = _turn("g1", 0, "kitchen")
    # Both panels use the same game support, while still selecting distinct
    # states. This is the estimand required by the paired game-cluster analysis.
    a3 = _turn("g1", 1, "hall")
    n1 = _turn("g3", 0, "kitchen")
    n2 = _turn("g4", 0, "hall")
    score_rows, metadata, attrition = build_m3_panel(
        [a1, a3, n1, n2],
        {"A1": [_source(a1)], "A3": [_source(a3)]},
        {"A1": {a1.state.state_hash}, "A3": {a3.state.state_hash}},
        experiments_by_group={"A1": "random", "A3": "entropy"},
        neighbors=2,
        minimum_neighbors=2,
    )
    assert not attrition
    assert {row["group"] for row in metadata} == {"A1", "A3"}
    assert all(row["n_neighbors"] == 2 for row in metadata)
    assert all(len(rows) == 3 for rows in score_rows.values())
    selected = {a1.state.state_hash, a3.state.state_hash}
    assert all(
        row["target_state_hash"] not in selected
        for rows in score_rows.values()
        for row in rows
        if row["target_kind"] == "neighbor"
    )

    base_by_panel = {}
    updated_by_cell = {}
    for group, rows in score_rows.items():
        base = []
        for row in rows:
            identity = {
                key: row[key]
                for key in [
                    "score_id",
                    "source_id",
                    "group",
                    "target_kind",
                    "source_state_hash",
                    "target_state_hash",
                    "teacher_action",
                    "panel_group",
                    "panel_experiment",
                ]
            }
            base.append(
                {
                    **identity,
                    "checkpoint_group": "BASE",
                    "checkpoint_experiment": None,
                    "mean_target_log_probability": -2.0,
                    "target_tokens": 2,
                }
            )
        base_by_panel[group] = base
        for checkpoint, experiment in {"A1": "random", "A3": "entropy"}.items():
            updated = []
            for row in rows:
                identity = {
                    key: row[key]
                    for key in [
                        "score_id",
                        "source_id",
                        "group",
                        "target_kind",
                        "source_state_hash",
                        "target_state_hash",
                        "teacher_action",
                        "panel_group",
                        "panel_experiment",
                    ]
                }
                gain = 1.0 if row["target_kind"] == "self" else 0.25
                updated.append(
                    {
                        **identity,
                        "checkpoint_group": checkpoint,
                        "checkpoint_experiment": experiment,
                        "mean_target_log_probability": -2.0 + gain,
                        "target_tokens": 2,
                    }
                )
            updated_by_cell[(checkpoint, group)] = updated
    assembled = assemble_m3_rows(metadata, base_by_panel, updated_by_cell)
    assert len(assembled) == 4
    assert {
        (row["checkpoint_group"], row["panel_group"]) for row in assembled
    } == {("A1", "A1"), ("A1", "A3"), ("A3", "A1"), ("A3", "A3")}
    assert all(row["S_i"] == 1.0 for row in assembled)
    assert all(row["T_i"] == 0.25 for row in assembled)


def test_m3_panel_drops_sources_outside_cross_panel_common_game_support():
    a1_g1 = _turn("g1", 0, "one")
    a1_g2 = _turn("g2", 0, "two")
    a3_g1 = _turn("g1", 1, "three")
    n1 = _turn("g3", 0, "neighbor one")
    n2 = _turn("g4", 0, "neighbor two")
    score_rows, metadata, attrition = build_m3_panel(
        [a1_g1, a1_g2, a3_g1, n1, n2],
        {"A1": [_source(a1_g1), _source(a1_g2)], "A3": [_source(a3_g1)]},
        {
            "A1": {a1_g1.state.state_hash, a1_g2.state.state_hash},
            "A3": {a3_g1.state.state_hash},
        },
        experiments_by_group={"A1": "random", "A3": "entropy"},
        neighbors=2,
        minimum_neighbors=1,
    )
    assert {row["game_id"] for row in metadata} == {"g1"}
    assert all(row["game_id"] == "g1" for row in metadata)
    assert all(
        row["source_id"] in {item["source_id"] for item in metadata}
        for rows in score_rows.values()
        for row in rows
    )
    assert any(
        row["reason"] == "outside_cross_panel_common_game_support"
        and row["source_state_hash"] == a1_g2.state.state_hash
        for row in attrition
    )


def test_m3_panel_records_attrition_instead_of_zero_filling_undefined_transfer():
    a1 = _turn("g1", 0, "one")
    a3 = _turn("g2", 0, "two")
    score_rows, metadata, attrition = build_m3_panel(
        [a1, a3],
        {"A1": [_source(a1)], "A3": [_source(a3)]},
        {"A1": {a1.state.state_hash}, "A3": {a3.state.state_hash}},
        experiments_by_group={"A1": "random", "A3": "entropy"},
        neighbors=2,
        minimum_neighbors=1,
    )
    assert metadata == []
    assert all(not rows for rows in score_rows.values())
    assert len(attrition) == 2


def test_m3_checkpoint_must_bind_exact_training_split_source():
    training = {
        "training_inputs": {
            "train_files": {"kind": "file", "sha256": "train"},
            "train_audit": {"kind": "file", "sha256": "audit"},
            "experiment_config": {"kind": "file", "sha256": "config"},
        },
        "arm_contract": {"teacher_samples_per_state_N": 1},
    }
    source = {
        "data_role": "actual_training_split_examples",
        "sha256": "train",
        "audit_sha256": "audit",
        "experiment_config": {"sha256": "config"},
    }
    assert _validate_checkpoint_source_binding(training, source) == {
        "training_data_sha256": "train",
        "training_audit_sha256": "audit",
        "experiment_config_sha256": "config",
    }
    training["training_inputs"]["train_files"]["sha256"] = "other"
    try:
        _validate_checkpoint_source_binding(training, source)
    except ValueError as error:
        assert "exact frozen source" in str(error)
    else:
        raise AssertionError("same experiment name must not substitute for source bytes")


def test_m3_scoring_rejects_tokenizer_distinct_from_training_base():
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        base = tmp_path / "base"
        other = tmp_path / "other-tokenizer"
        base.mkdir()
        other.mkdir()
        assert _resolve_training_tokenizer_path(base, None) == base
        assert _resolve_training_tokenizer_path(base, str(base)) == base
        try:
            _resolve_training_tokenizer_path(base, str(other))
        except ValueError as error:
            assert "exact base-model directory" in str(error)
        else:
            raise AssertionError("M3 must not score with a non-training tokenizer")
