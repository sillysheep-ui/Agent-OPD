from omniopd.io import correction_from_dict, record_game_id


def test_exclusion_game_id_is_read_from_canonical_nested_state():
    assert record_game_id({"state": {"game_id": "g/canonical"}}) == "g/canonical"
    assert record_game_id({"gamefile": "g/legacy"}) == "g/legacy"


def test_serialized_state_hash_is_verified():
    data = {
        "state": {
            "task": "t",
            "observation": "o",
            "admissible_actions": ["look"],
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ],
            "game_id": "g",
            "task_type": "x",
            "turn_index": 0,
            "state_hash": "tampered",
        },
        "student": {
            "raw": "Action: look",
            "executed_action": "look",
            "valid": True,
            "had_action_marker": True,
            "failure_reason": None,
            "api_calls": 1,
        },
        "teacher_samples": [],
        "selection_policy": "random",
        "inclusion_probability": 1.0,
        "teacher_calls": 0,
    }
    try:
        correction_from_dict(data)
    except ValueError as error:
        assert "state hash mismatch" in str(error)
    else:
        raise AssertionError("a modified serialized state must be rejected")
