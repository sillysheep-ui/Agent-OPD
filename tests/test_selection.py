import random

from omniopd.schema import ActionSample, AgentState, RolloutTurn
from omniopd.selection import (
    admissible_entropy,
    select_top_score,
    select_uniform,
    select_uniform_nested,
)


def _turn(index):
    state = AgentState("t", f"o{index}", ("look",), ({"role": "system", "content": "s"}, {"role": "user", "content": "u"}), "g", "x", index)
    action = ActionSample("Action: look", "look", True, True, None)
    return RolloutTurn(state, action)


def test_uniform_selection_records_inclusion_probability():
    selected = select_uniform([_turn(i) for i in range(10)], 3, random.Random(7))
    assert len(selected) == 3
    assert all(item.inclusion_probability == 0.3 for item in selected)


def test_entropy_is_stable_for_large_scores():
    value = admissible_entropy([10_000.0, 10_000.0])
    assert abs(value - 0.6931471805599453) < 1e-12


def test_deterministic_top_k_does_not_emit_nonstandard_json_nan_probability():
    selected = select_top_score([_turn(0), _turn(1)], [0.1, 0.9], 1, policy="entropy_top_k")
    assert selected[0].turn.state.turn_index == 1
    assert selected[0].inclusion_probability is None


def test_uniform_priority_selection_is_nested_across_breadth_depth():
    turns = [_turn(index) for index in range(8)]
    depth = select_uniform_nested(turns, 1, seed=42, game_id="g")
    breadth = select_uniform_nested(turns, 3, seed=42, game_id="g")
    assert {item.turn.state.state_hash for item in depth} <= {
        item.turn.state.state_hash for item in breadth
    }
    assert all(item.inclusion_probability == 1 / 8 for item in depth)
    assert all(item.inclusion_probability == 3 / 8 for item in breadth)
