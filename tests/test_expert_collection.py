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


def test_stratified_split_covers_every_task_family():
    from omniopd.dataset import split_rows_by_game_stratified

    rows = []
    for task_type in ("pick_and_place_simple", "look_at_obj_in_light"):
        for game_index in range(4):
            for step in range(2):
                rows.append(
                    {
                        "game_id": f"{task_type}-{game_index}",
                        "task_type": task_type,
                        "episode_step": step,
                    }
                )
    train, validation, metadata = split_rows_by_game_stratified(
        rows,
        val_fraction=0.25,
        rng_seed=11,
        stratum_key=lambda row: row["task_type"],
        game_key=lambda row: row["game_id"],
    )
    assert len(train) + len(validation) == len(rows)
    assert metadata["split_unit"] == "game_within_stratum"
    for task_type, stats in metadata["strata"].items():
        assert stats["validation_games"], task_type
        assert stats["train_games"], task_type
        assert not set(stats["validation_games"]) & set(stats["train_games"])
    assert {row["task_type"] for row in validation} == {
        "pick_and_place_simple",
        "look_at_obj_in_light",
    }


def test_stratified_split_refuses_a_single_game_family():
    from omniopd.dataset import split_rows_by_game_stratified

    rows = [
        {"game_id": "solo-a", "task_type": "solo"},
        {"game_id": "pair-a", "task_type": "pair"},
        {"game_id": "pair-b", "task_type": "pair"},
    ]
    try:
        split_rows_by_game_stratified(
            rows,
            val_fraction=0.5,
            rng_seed=3,
            stratum_key=lambda row: row["task_type"],
            game_key=lambda row: row["game_id"],
        )
    except ValueError as error:
        assert "solo" in str(error)
    else:
        raise AssertionError("a single-game family cannot fill both sides")


def test_collector_defaults_to_ten_solved_episodes_per_family():
    import sys

    from scripts.collect_expert_sft import parse_args

    argv = sys.argv
    sys.argv = ["collect_expert_sft.py", "--env-config", "c.yaml", "--tokenizer", "t", "--output", "o"]
    try:
        args = parse_args()
    finally:
        sys.argv = argv
    assert args.solved_per_task_type == 10
    assert args.max_attempts_per_task_type == 50
    assert args.split == "train"


def test_held_out_games_preserves_first_seen_order(tmp_path=None):
    import json
    import tempfile
    from pathlib import Path

    from scripts.evaluate_coldstart_sft import held_out_games

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "val.jsonl"
        rows = [
            {"game_id": "g-b", "task_type": "look"},
            {"game_id": "g-b", "task_type": "look"},
            {"game_id": "g-a", "task_type": "pick"},
            {"game_id": "g-c", "task_type": "look"},
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        games = held_out_games(path)
    assert games == {"look": ["g-b", "g-c"], "pick": ["g-a"]}


def test_game_list_cache_is_reused_and_split_checked():
    import json
    import tempfile
    from pathlib import Path

    from scripts.collect_expert_sft import resolve_game_list

    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "train_games.json"
        cache.write_text(
            json.dumps({"split": "train", "games": ["/a/game.tw-pddl", "/b/game.tw-pddl"]}),
            encoding="utf-8",
        )
        games, source = resolve_game_list(
            config={}, split="train", cache_path=cache, task_types=("pick_and_place_simple",)
        )
        assert games == ["/a/game.tw-pddl", "/b/game.tw-pddl"]
        assert source == "cache"

        try:
            resolve_game_list(
                config={},
                split="eval_out_of_distribution",
                cache_path=cache,
                task_types=("pick_and_place_simple",),
            )
        except SystemExit as error:
            assert "split" in str(error)
        else:
            raise AssertionError("a cache for another split must not be reused")


def test_game_list_cache_written_on_miss():
    import json
    import tempfile
    from pathlib import Path

    from scripts import collect_expert_sft

    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "train_games.json"
        original = collect_expert_sft.enumerate_split_games
        collect_expert_sft.enumerate_split_games = lambda **kwargs: ["/x/game.tw-pddl"]
        try:
            games, source = collect_expert_sft.resolve_game_list(
                config={"dataset": {"data_path": "/tmp"}},
                split="train",
                cache_path=cache,
                task_types=("pick_and_place_simple",),
            )
        finally:
            collect_expert_sft.enumerate_split_games = original
        assert games == ["/x/game.tw-pddl"]
        assert source == "alfworld_environment"
        assert json.loads(cache.read_text(encoding="utf-8")) == {
            "split": "train",
            "games": ["/x/game.tw-pddl"],
        }


def test_fast_enumerator_reads_one_directory_level_at_a_time():
    import tempfile
    from pathlib import Path

    from scripts.collect_expert_sft import enumerate_split_games

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        wanted = root / "pick_and_place_simple-Apple-None-Desk-1" / "trial_a"
        wanted.mkdir(parents=True)
        (wanted / "game.tw-pddl").write_text("{}", encoding="utf-8")
        other = root / "look_at_obj_in_light-DeskLamp-201" / "trial_b"
        other.mkdir(parents=True)
        (other / "game.tw-pddl").write_text("{}", encoding="utf-8")
        skipped = root / "pick_and_place_simple-movable-2" / "trial_c"
        skipped.mkdir(parents=True)
        (skipped / "game.tw-pddl").write_text("{}", encoding="utf-8")

        games = enumerate_split_games(
            data_path=root, task_types=("pick_and_place_simple",)
        )
    assert games == [str(wanted / "game.tw-pddl")]


def test_student_prompt_versions_resolve_and_record():
    from omniopd.prompts import (
        STUDENT_SYSTEM_PROMPT,
        STUDENT_SYSTEM_PROMPTS,
        resolve_student_prompt,
    )

    assert resolve_student_prompt() == ("v1", STUDENT_SYSTEM_PROMPT)
    name, text = resolve_student_prompt("v2")
    assert name == "v2"
    assert text in STUDENT_SYSTEM_PROMPTS.values()
    assert text != STUDENT_SYSTEM_PROMPT
    assert "Track the task state" in text
    try:
        resolve_student_prompt("v3")
    except ValueError as error:
        assert "unknown Student prompt version" in str(error)
    else:
        raise AssertionError("an unknown prompt version must fail closed")


def test_expert_rows_follow_the_requested_prompt_version():
    from omniopd.prompts import resolve_student_prompt
    from scripts.build_expert_sft_data import build_rows

    _, prompt_text = resolve_student_prompt("v2")
    rows, _ = build_rows(
        [_expert_episode("game-a", ["look"])], student_prompt=prompt_text
    )
    assert rows[0]["messages"][0] == {"role": "system", "content": prompt_text}


def test_dataset_checks_the_configured_prompt_version():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        from unittest import SkipTest

        raise SkipTest("optional training dependencies are not installed")

    import json
    import tempfile
    from pathlib import Path

    from omniopd.prompts import resolve_student_prompt
    from omniopd.torch_dataset import FinalTurnActionDataset

    _, prompt_text = resolve_student_prompt("v2")
    row = {
        "messages": [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "Action: look"},
        ],
        "state_weight": 1.0,
        "state_hash": "h",
        "game_id": "g",
        "turn_index": 0,
        "target_sample_index": 0,
        "target_action": "look",
        "target_source": "alfworld_handcoded_expert",
        "weighting_mode": "game_state_mean",
        "protocol_version": "omniopd-v1",
        "enable_thinking": False,
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        dataset = FinalTurnActionDataset(
            files=path, tokenizer=FakeTokenizer(), max_length=10, config={"student_prompt": "v2"}
        )
        assert dataset.student_prompt_name == "v2"
        try:
            FinalTurnActionDataset(files=path, tokenizer=FakeTokenizer(), max_length=10)
        except ValueError as error:
            assert "Student-context" in str(error)
        else:
            raise AssertionError("a v2 row must not pass the default v1 dataset check")
