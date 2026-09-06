from omniopd.analysis.sage import (
    build_sage_judge_messages,
    design_weighted_game_balanced_indicator,
    deterministic_or_judged_label,
    disagreement,
    game_cluster_bootstrap_design_ratio,
    normalize_teacher_sample,
    parse_sage_label,
    state_level_disagreement,
)
from omniopd.schema import AgentState


def test_sage_normalizes_new_dict_schema_before_disagreement():
    sample = {"teacher_action": "go", "teacher_valid": True, "teacher_raw": "Action: go"}
    assert normalize_teacher_sample(sample)["action"] == "go"
    assert disagreement("look", sample) == 1


def test_invalid_teacher_disagreement_remains_missing():
    assert disagreement("look", {"teacher_action": "look", "teacher_valid": False}) is None


def test_state_disagreement_is_any_valid_teacher_draw():
    samples = [
        {"teacher_action": "look", "teacher_valid": True},
        {"teacher_action": "go", "teacher_valid": True},
    ]
    assert state_level_disagreement("look", samples) == 1
    assert state_level_disagreement(
        "look", [{"teacher_action": "go", "teacher_valid": False}]
    ) is None


def test_technical_state_reaches_deterministic_strong_branch():
    assert deterministic_or_judged_label(student_valid=False, judge_label=None) == "Strong"


def test_failed_judge_call_remains_missing_instead_of_being_imputed():
    assert deterministic_or_judged_label(student_valid=True, judge_label=None) is None


def test_sage_population_summary_uses_known_inclusion_probabilities():
    rows = [
        {"game_id": "g1", "strong": 1, "inclusion_probability": 0.5},
        {"game_id": "g1", "strong": 0, "inclusion_probability": 0.5},
        {"game_id": "g2", "strong": 0, "inclusion_probability": 0.5},
        {"game_id": "g2", "strong": 0, "inclusion_probability": 0.5},
    ]
    result = design_weighted_game_balanced_indicator(
        rows,
        indicator_key="strong",
        population_states_by_game={"g1": 4, "g2": 4},
    )
    assert result.estimate == 0.25


def test_sage_rejects_top_k_without_design_probability():
    try:
        design_weighted_game_balanced_indicator(
            [{"game_id": "g", "strong": 1, "inclusion_probability": float("nan")}],
            indicator_key="strong",
            population_states_by_game={"g": 2},
        )
    except ValueError as error:
        assert "top-k" in str(error)
    else:
        raise AssertionError("top-k samples do not identify the overall population proportion")


def test_sage_conditionals_have_game_cluster_bootstrap_and_reject_missing():
    rows = [
        {"game_id": game, "num": value, "den": 1, "inclusion_probability": 1.0}
        for game, value in [("g1", 1), ("g2", 0), ("g3", 1)]
    ]
    result = game_cluster_bootstrap_design_ratio(
        rows,
        numerator_key="num",
        denominator_key="den",
        population_states_by_game={"g1": 1, "g2": 1, "g3": 1},
        replicates=100,
        rng_seed=1,
    )
    assert result.estimate == 2 / 3
    broken = [dict(rows[0], num=None), *rows[1:]]
    try:
        game_cluster_bootstrap_design_ratio(
            broken,
            numerator_key="num",
            denominator_key="den",
            population_states_by_game={"g1": 1, "g2": 1, "g3": 1},
            replicates=10,
        )
    except ValueError as error:
        assert "nonidentified" in str(error)
    else:
        raise AssertionError("missing SAGE outcomes must not be silently imputed")


def test_sage_judge_prompt_is_blind_to_teacher_outputs_and_parses_exact_labels():
    state = AgentState(
        "task",
        "observation",
        ("look", "go"),
        (
            {"role": "system", "content": "student system"},
            {"role": "user", "content": "visible state"},
        ),
        "game",
        "type",
        0,
    )
    messages = build_sage_judge_messages(state, "look")
    serialized = str(messages)
    assert "student system" not in serialized
    assert "teacher_action" not in serialized
    assert "entropy" not in serialized
    assert parse_sage_label("Weak") == "Weak"
    assert parse_sage_label("Reasoning... Weak") is None
