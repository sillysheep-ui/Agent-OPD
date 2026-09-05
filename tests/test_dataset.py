from collections import defaultdict

from omniopd.dataset import build_training_rows, split_rows_by_game
from omniopd.schema import ActionSample, AgentState, CorrectionRecord


def _record(game, turn, actions):
    state = AgentState(
        "task",
        "obs",
        ("look", "go"),
        ({"role": "system", "content": "s"}, {"role": "user", "content": "u"}),
        game,
        "type",
        turn,
    )
    student = ActionSample("Action: look", "look", True, True, None)
    teacher = [ActionSample(f"Action: {a}", a, True, True, None) for a in actions]
    return CorrectionRecord(state, student, teacher, "random", 1.0, len(teacher))


def test_game_state_weights_sum_to_one_per_game_and_state_splits_across_k():
    rows = build_training_rows(
        [_record("g1", 0, ["go", "look"]), _record("g1", 1, ["go"]), _record("g2", 0, ["go"])],
        weighting="game_state_mean",
    )
    per_game = defaultdict(float)
    per_state = defaultdict(float)
    for row in rows:
        per_game[row.game_id] += row.state_weight
        per_state[(row.game_id, row.turn_index)] += row.state_weight
        assert row.messages[-1]["role"] == "assistant"
    assert per_game == {"g1": 1.0, "g2": 1.0}
    assert per_state[("g1", 0)] == 0.5
    assert per_state[("g1", 1)] == 0.5


def test_split_keeps_games_and_multi_action_states_together():
    rows = build_training_rows(
        [_record("g1", 0, ["go", "look"]), _record("g2", 0, ["go"]), _record("g3", 0, ["go"])],
        weighting="state_mean",
    )
    train, validation, metadata = split_rows_by_game(rows, val_fraction=1 / 3, rng_seed=9)
    train_games = {row.game_id for row in train}
    validation_games = {row.game_id for row in validation}
    assert train_games.isdisjoint(validation_games)
    assert metadata["split_unit"] == "game"
    for game in {"g1", "g2", "g3"}:
        assert game in train_games or game in validation_games
