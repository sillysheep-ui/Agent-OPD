import tempfile
import json
from pathlib import Path

from omniopd.prompts import STUDENT_SYSTEM_PROMPT


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking=False
    ):
        ids = [1]
        for message in messages:
            if message["role"] == "system":
                ids += [10]
            elif message["role"] == "user":
                ids += [20]
            else:
                ids += [30] + ([31] if message["content"] else []) + [32]
        if add_generation_prompt:
            ids += [30]
        return ids


def test_verl_dataset_emits_canonical_mask_and_zero_weight_padding_index():
    try:
        import pandas as pd
        import torch  # noqa: F401
    except ModuleNotFoundError:
        from unittest import SkipTest

        raise SkipTest("optional training dependencies are not installed")

    from omniopd.torch_dataset import FinalTurnActionDataset

    messages = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "Action: look"},
    ]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.parquet"
        pd.DataFrame(
            [
                {
                    "messages": messages,
                    "state_weight": 1.0,
                    "state_hash": "hash",
                    "game_id": "game",
                    "turn_index": 0,
                    "teacher_sample_index": 0,
                    "teacher_action": "look",
                    "weighting_mode": "game_state_mean",
                    "protocol_version": "omniopd-v1",
                    "enable_thinking": False,
                }
            ]
        ).to_parquet(path, index=False)
        dataset = FinalTurnActionDataset(
            parquet_files=[path], tokenizer=FakeTokenizer(), max_length=10
        )
        real = dataset[0]
        padded_duplicate = dataset[(0, 0.0)]
        assert real["input_ids"].shape == (10,)
        assert real["target_token_mask"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        assert real["state_weight"].item() == 1.0
        assert padded_duplicate["state_weight"].item() == 0.0
        assert real["sampling_multiplier"].item() == 1.0
        assert padded_duplicate["sampling_multiplier"].item() == 0.0
        assert dataset.objective_weight_sum == 1.0
        assert dataset.objective_population_size == 1
        assert set(real) == {
            "input_ids",
            "attention_mask",
            "position_ids",
            "target_token_mask",
            "state_weight",
            "sampling_multiplier",
        }
        try:
            FinalTurnActionDataset(
                parquet_files=[path],
                tokenizer=FakeTokenizer(),
                max_length=10,
                truncation="left",
            )
        except ValueError as error:
            assert "TaskPreservingTruncator" in str(error)
        else:
            raise AssertionError("training must not silently left-truncate the state")


def test_verl_dataset_accepts_the_builder_jsonl_format():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        from unittest import SkipTest

        raise SkipTest("optional training dependencies are not installed")

    from omniopd.torch_dataset import FinalTurnActionDataset

    row = {
        "messages": [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "Action: look"},
        ],
        "state_weight": 1.0,
        "state_hash": "hash-jsonl",
        "game_id": "game-jsonl",
        "turn_index": 0,
        "teacher_sample_index": 0,
        "teacher_action": "look",
        "weighting_mode": "game_state_mean",
        "protocol_version": "omniopd-v1",
        "enable_thinking": False,
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        dataset = FinalTurnActionDataset(
            files=path,
            tokenizer=FakeTokenizer(),
            max_length=10,
        )
        assert len(dataset) == 1


def test_verl_dataset_accepts_generic_expert_targets_without_teacher_aliases():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        from unittest import SkipTest

        raise SkipTest("optional training dependencies are not installed")

    from omniopd.torch_dataset import FinalTurnActionDataset

    row = {
        "messages": [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "Action: look"},
        ],
        "state_weight": 1.0,
        "state_hash": "expert-hash",
        "game_id": "expert-game",
        "turn_index": 0,
        "target_sample_index": 0,
        "target_action": "look",
        "target_source": "alfworld_handcoded_expert",
        "weighting_mode": "game_state_mean",
        "protocol_version": "omniopd-v1",
        "enable_thinking": False,
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "expert.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        dataset = FinalTurnActionDataset(files=path, tokenizer=FakeTokenizer(), max_length=10)
        assert len(dataset) == 1
        assert dataset.objective_weight_sum == 1.0
