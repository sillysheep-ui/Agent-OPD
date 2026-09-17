#!/usr/bin/env python3
"""CPU contract check for the custom veRL OPD worker on one real state."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.io import read_jsonl, rollout_turn_from_dict
from omniopd.opd_adapter import build_fixed_pool_opd_prompts
from omniopd.tokenization import apply_chat_template_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-pool", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--hf-smoke", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from omniopd.verl_opd import OmniOPDAgentLoopWorker
    from verl.trainer.distillation.losses import distillation_loss
    from verl.workers.utils.padding import no_padding_2_padding
    from verl.workers.config import DistillationLossConfig
    from tensordict import TensorDict
    import torch

    turn = rollout_turn_from_dict(next(read_jsonl(args.state_pool)))
    reference = json.loads(args.hf_smoke.read_text(encoding="utf-8"))
    if reference["state_hash"] != turn.state.state_hash:
        raise ValueError("worker check state differs from reference smoke")
    student = AutoTokenizer.from_pretrained(
        str(args.student_model), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    teacher = AutoTokenizer.from_pretrained(
        str(args.teacher_model), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    selection = {
        "protocol_version": "omniopd-v1",
        "state_hash": turn.state.state_hash,
        "game_id": turn.state.game_id,
        "turn_index": turn.state.turn_index,
        "selection_policy": "uniform_per_game_nested_v1",
    }
    row = build_fixed_pool_opd_prompts([turn], [selection])[0]
    prompt_ids = apply_chat_template_ids(
        student, row["prompt"], add_generation_prompt=True, enable_thinking=False
    )
    teacher_prompt_ids = apply_chat_template_ids(
        teacher, row["extra_info"]["teacher_prompt"],
        add_generation_prompt=True, enable_thinking=False,
    )
    content_ids = [int(token) for token in reference["student_action_token_ids"]]
    response_ids = content_ids + [student.eos_token_id]
    expected_scores = [float(value) for value in reference["teacher_sampled_token_logprobs"]]

    class FakeTeacherManager:
        teacher_model_configs = {
            "teacher_model": SimpleNamespace(
                model_path=str(args.teacher_model),
                inference=SimpleNamespace(max_model_len=2048),
            )
        }

        async def compute_teacher_logprobs_single(self, *, sequence_ids, routing_key):
            if routing_key != "teacher_model" or sequence_ids != teacher_prompt_ids + content_ids:
                raise AssertionError("worker requested a wrong Teacher sequence")
            ids = [[token] for token in sequence_ids[1:]] + [[0]]
            values = [[0.0] for _ in sequence_ids]
            for index, score in enumerate(expected_scores, len(teacher_prompt_ids) - 1):
                values[index] = [score]
            return torch.tensor(ids, dtype=torch.int32), torch.tensor(values)

    worker = object.__new__(OmniOPDAgentLoopWorker)
    worker.distillation_enabled = True
    worker.teacher_server_manager = FakeTeacherManager()
    worker.tokenizer = student
    output = SimpleNamespace(extra_fields={})
    asyncio.run(
        worker._compute_teacher_logprobs(
            output,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            validate=False,
            sample_kwargs={"raw_prompt": row["prompt"], "extra_info": row["extra_info"]},
        )
    )
    sequence_length = len(prompt_ids) + len(response_ids)
    if output.extra_fields["teacher_logprobs"].shape != (sequence_length, 1):
        raise AssertionError("worker returned wrong Teacher tensor shape")
    data = TensorDict(
        {
            "prompts": torch.tensor([prompt_ids]),
            "responses": torch.tensor([response_ids]),
            "attention_mask": torch.ones((1, sequence_length), dtype=torch.int64),
        },
        batch_size=[1],
    )
    sliced = no_padding_2_padding(output.extra_fields["teacher_logprobs"], data)
    actual_scores = sliced.squeeze(0).squeeze(-1).tolist()
    if any(abs(a - b) > 1e-6 for a, b in zip(actual_scores[:-1], expected_scores)):
        raise AssertionError("worker response scores disagree with the HF reference")
    if len(actual_scores) != len(response_ids) or actual_scores[-1] != 0.0:
        raise AssertionError("worker did not mask the Student EOS score")

    # Exercise veRL's real sampled-token KL estimator and response mask. The
    # artificial Student scores straddle the fixed Teacher scores so the
    # expected gradient signs are unambiguous; EOS must have zero gradient.
    offsets = torch.tensor([0, sequence_length], dtype=torch.int64)
    nested_teacher = torch.nested.nested_tensor_from_jagged(
        values=output.extra_fields["teacher_logprobs"], offsets=offsets
    )
    student_scores = torch.tensor(
        [0.0] * (len(prompt_ids) - 1)
        + [score + (0.5 if index % 2 == 0 else -0.5) for index, score in enumerate(expected_scores)]
        + [100.0, 0.0],
        requires_grad=True,
    )
    if student_scores.numel() != sequence_length:
        raise AssertionError("Student score fixture has the wrong width")
    loss_data = TensorDict(
        {
            "prompts": data["prompts"],
            "responses": data["responses"],
            "attention_mask": data["attention_mask"],
            "response_mask": torch.tensor([[1] * len(content_ids) + [0]], dtype=torch.int64),
            "teacher_logprobs": nested_teacher,
        },
        batch_size=[1],
    )
    loss_config = DistillationLossConfig(
        loss_mode="k3", use_task_rewards=False, use_policy_gradient=False
    )
    loss, _ = distillation_loss(
        config=SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}),
        distillation_config=SimpleNamespace(distillation_loss=loss_config),
        model_output={"log_probs": student_scores},
        data=loss_data,
    )
    loss.backward()
    response_gradients = student_scores.grad[len(prompt_ids) - 1 : -1].tolist()
    if any(
        gradient <= 0 if index % 2 == 0 else gradient >= 0
        for index, gradient in enumerate(response_gradients[:-1])
    ):
        raise AssertionError("veRL sampled-token KL gradients have unexpected directions")
    if response_gradients[-1] != 0.0 or student_scores.grad[-1].item() != 0.0:
        raise AssertionError("masked EOS or dummy final row received a distillation gradient")
    if not torch.isfinite(loss).item():
        raise AssertionError("veRL sampled-token KL loss is not finite")
    print("real-state OPD worker and veRL sampled-token KL gradient contract: OK")


if __name__ == "__main__":
    main()
