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
