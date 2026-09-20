#!/usr/bin/env python3
"""Check OPD remapping against veRL v0.8.0 extraction and response slicing."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.opd_adapter import remap_teacher_scores_to_student_layout


def main() -> None:
    import torch
    from tensordict import TensorDict
    from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs
    from verl.workers.utils.padding import no_padding_2_padding

    teacher_sequence = [3, 4, 5, 6, 7]
    fake_vllm_output = SimpleNamespace(
        prompt_logprobs=[None]
        + [
            {token: SimpleNamespace(logprob=float(-index))}
            for index, token in enumerate(teacher_sequence[1:], 1)
        ]
    )
    extracted = {}
    extract_prompt_logprobs(fake_vllm_output, 0, extracted)
    aligned_ids, aligned_scores = remap_teacher_scores_to_student_layout(
        student_prompt_ids=[1, 2],
        teacher_prompt_ids=[3, 4, 5],
        student_response_ids=[6, 7],
        teacher_scored_ids=extracted["prompt_ids"],
        teacher_scored_logprobs=extracted["prompt_logprobs"],
        pad_token_id=0,
    )
    # veRL's third response token is EOS, deliberately masked and unscored.
    aligned_ids.append([0])
    aligned_scores.append([0.0])
    if aligned_ids != [[0], [6], [7], [0], [0]]:
        raise AssertionError(f"shifted Teacher IDs differ: {aligned_ids}")
    data = TensorDict(
        {
            "prompts": torch.tensor([[1, 2]]),
            "responses": torch.tensor([[6, 7, 99]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.int64),
        },
        batch_size=[1],
    )
    response_scores = no_padding_2_padding(torch.tensor(aligned_scores), data)
    actual = response_scores.squeeze(0).squeeze(-1).tolist()
    if actual != [-3.0, -4.0, 0.0]:
        raise AssertionError(f"veRL response slicing misaligned Teacher scores: {actual}")
    print("veRL Teacher extraction -> OmniOPD remap -> response slice: OK")


if __name__ == "__main__":
    main()
