from scripts.collect_expert_sft import select_games, task_family_from_path


def _game(task_type: str, index: int) -> str:
    return (
        f"/data/json_2.1.1/train/{task_type}-Apple-None-Desk-{index}/"
        f"trial_T20190908_125747_{index}/game.tw-pddl"
    )


def test_task_family_is_read_from_the_trial_directory_name():
    game = _game("pick_heat_then_place_in_recep", 1)
    assert task_family_from_path(game) == "pick_heat_then_place_in_recep"


def test_task_family_falls_back_to_the_family_directory():
    assert task_family_from_path("/x/train/look_at_obj_in_light-Desk/game.tw-pddl") == (
        "look_at_obj_in_light"
    )


def test_selection_is_stable_and_balanced_per_task_type():
    games = [
        _game(task_type, index)
        for task_type in ("pick_and_place_simple", "look_at_obj_in_light")
        for index in range(5)
    ]
    first = select_games(
        games,
        task_types=("pick_and_place_simple", "look_at_obj_in_light"),
        games_per_task_type=2,
        seed=42,
    )
    second = select_games(
        list(reversed(games)),
        task_types=("pick_and_place_simple", "look_at_obj_in_light"),
        games_per_task_type=2,
        seed=42,
    )
    assert first == second
    assert len(first) == 4
    per_type = {}
    for game in first:
        per_type.setdefault(task_family_from_path(game), []).append(game)
    assert sorted(len(items) for items in per_type.values()) == [2, 2]


def test_selection_refuses_an_underfilled_task_type():
    games = [_game("pick_and_place_simple", index) for index in range(2)]
    try:
        select_games(
            games,
            task_types=("pick_and_place_simple", "look_at_obj_in_light"),
            games_per_task_type=2,
            seed=42,
        )
    except SystemExit as error:
        assert "look_at_obj_in_light" in str(error)
    else:
        raise AssertionError("an underfilled task type must stop the collection")


def _expert_turn(game_id: str, turn_index: int, action: str, *, valid: bool = True) -> dict:
    from omniopd.prompts import STUDENT_SYSTEM_PROMPT
    from omniopd.schema import ActionSample, AgentState

    state = AgentState(
        task="put a hot apple in countertop",
        observation=f"observation {turn_index}",
        admissible_actions=(action, "look"),
        messages=(
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"observation {turn_index}"},
        ),
        game_id=game_id,
        task_type="pick_heat_then_place_in_recep",
        turn_index=turn_index,
    )
    sample = ActionSample(
        raw=f"Action: {action}",
        executed_action=action,
        valid=valid,
        had_action_marker=True,
        failure_reason=None if valid else "not_admissible",
    )
    return {"state": state.to_dict(), "student": sample.to_dict()}


def _expert_episode(game_id: str, actions: list[str], *, won: bool = True) -> dict:
    return {
        "game_id": game_id,
        "task_type": "pick_heat_then_place_in_recep",
        "environment_seed": 1,
        "won": won,
        "steps": len(actions),
        "invalid_turns": 0,
        "turns": [
            _expert_turn(game_id, index, action) for index, action in enumerate(actions)
        ],
    }


def test_expert_rows_weight_each_game_equally_and_drop_unsolved_episodes():
    import math

    from scripts.build_expert_sft_data import build_rows

    episodes = [
        _expert_episode("game-a", ["look", "go to countertop 1"]),
        _expert_episode("game-b", ["look"], won=False),
    ]
    rows, counts = build_rows(episodes)

    assert counts["episodes_kept"] == 1
    assert counts["episodes_dropped_unsolved"] == 1
    assert counts["turns_dropped_unsolved"] == 1
    assert len(rows) == 2
    assert {row["game_id"] for row in rows} == {"game-a"}
    assert math.isclose(sum(row["state_weight"] for row in rows), 1.0)
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": "Action: look"}
    assert rows[0]["target_action"] == "look"
    assert rows[0]["target_source"] == "alfworld_handcoded_expert"
    assert rows[0]["weighting_mode"] == "game_state_mean"
    assert rows[0]["enable_thinking"] is False


def test_expert_rows_refuse_an_invalid_action_inside_a_solved_episode():
    from scripts.build_expert_sft_data import build_rows

    episode = _expert_episode("game-a", ["look"])
    episode["turns"][0] = _expert_turn("game-a", 0, "look", valid=False)
    try:
        build_rows([episode])
    except ValueError as error:
        assert "invalid action" in str(error)
    else:
        raise AssertionError("an invalid expert action must not become a target")


def test_game_split_accepts_dict_rows_through_a_game_key():
    from omniopd.dataset import split_rows_by_game

    rows = [{"game_id": f"game-{index}"} for index in range(4)]
    train, validation, metadata = split_rows_by_game(
        rows,
        val_fraction=0.25,
        rng_seed=7,
        game_key=lambda row: row["game_id"],
    )
    assert len(train) + len(validation) == 4
    assert metadata["split_unit"] == "game"
    assert metadata["validation_games"] and metadata["train_games"]
