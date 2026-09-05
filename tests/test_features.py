from omniopd.features import state_diagnostic_features
from omniopd.schema import ActionSample, AgentState, RolloutTurn


def test_m2_features_are_derived_from_the_canonical_state_and_action():
    state = AgentState(
        task="task",
        observation="same",
        admissible_actions=("look", "go north"),
        messages=(
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ),
        game_id="g",
        task_type="pick_and_place",
        turn_index=5,
    )
    turn = RolloutTurn(
        state,
        ActionSample("", "look", False, False, "empty_content"),
    )
    features = state_diagnostic_features(
        turn, previous_observation="same", max_steps=50
    )
    assert features["task_type"] == "pick_and_place"
    assert features["action_type"] == "look"
    assert features["technical"] == 1
    assert features["empty_response"] == 1
    assert features["repeat_observation"] == 1
