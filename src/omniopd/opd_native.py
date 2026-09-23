"""Minimal veRL-native OPD for this project: two small overrides, nothing else.

Everything the run needs already exists in veRL (rollout, response mask,
normalisation, distillation losses, clamps, optimiser, checkpointing). Only two
things have to change for *this* model pair and this task, and both are
properties of the wiring rather than of the method:

1. The Teacher must be asked about the conversation rendered with **its own**
   chat template. veRL's stock worker hands the Teacher the Student's token ids
   (``agent_loop.py: _compute_teacher_logprobs``), which is only equivalent when
   both models share a template. Ours do not: the Student is an Instruct-2507
   release without a thinking branch, the Teacher is a hybrid-thinking release
   whose ``enable_thinking=False`` template inserts an empty thinking block.
   Measured: with the Student's rendering the Teacher's argmax at the first
   response token is ``<think>`` and it scores the token the Student produced
   about 21 nats lower, in 100% of sampled turns.

2. The task-reward slot must be filled. OPD supervises tokens, not outcomes, so
   the dataset carries no ``reward_model``; veRL only skips its reward loop when
   a rollout already carries a score (``agent_loop.py: _compute_score``). A
   plumbing zero is the honest value here, not invented reward data.

Nothing else in this file touches the loss, the mask or the optimiser.
"""

from __future__ import annotations

import logging
from typing import Any

import ray
import torch
from transformers import AutoTokenizer
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from .opd_native_layout import (
    render_prompt,
    teacher_messages,
    teacher_rows_for_student_layout,
)

logger = logging.getLogger(__name__)


class ZeroRewardSingleTurnAgentLoop(SingleTurnAgentLoop):
    """Stock rollout with the unused task-reward slot set to zero (see module doc)."""

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        output = await super().run(sampling_params, **kwargs)
        output.reward_score = 0.0
        return output


class TeacherTemplateAgentLoopWorker(AgentLoopWorker):
    """Stock worker that scores the Teacher on its own rendering of the state."""

    def _teacher_tokenizer(self):
        configs = self.teacher_server_manager.teacher_model_configs
        if len(configs) != 1:
            raise ValueError("this implementation supports exactly one frozen Teacher")
        if not hasattr(self, "_teacher_tokenizer_cache"):
            self._teacher_tokenizer_cache = AutoTokenizer.from_pretrained(
                configs[next(iter(configs))].model_path,
                use_fast=True,
                local_files_only=True,
                trust_remote_code=False,
            )
        return self._teacher_tokenizer_cache

    async def _compute_teacher_logprobs(
        self,
        output,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if validate or not self.distillation_enabled:
            return
        student_prompt = [int(token) for token in prompt_ids]
        response = [int(token) for token in response_ids]
        if not response:
            raise ValueError("Teacher scoring needs a non-empty Student response")
        teacher_prompt = render_prompt(self._teacher_tokenizer(), teacher_messages(sample_kwargs))
        teacher_key = next(iter(self.teacher_server_manager.teacher_model_configs))
        teacher_ids, teacher_logprobs = (
            await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=teacher_prompt + response,
                routing_key=teacher_key,
            )
        )
        rows, scores = teacher_rows_for_student_layout(
            student_prompt=student_prompt,
            teacher_prompt=teacher_prompt,
            response=response,
            teacher_ids=teacher_ids.tolist(),
            teacher_logprobs=teacher_logprobs.tolist(),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output.extra_fields["teacher_ids"] = torch.tensor(rows, dtype=torch.int32)
        output.extra_fields["teacher_logprobs"] = torch.tensor(scores, dtype=torch.float32)
        if not getattr(self, "_teacher_hook_logged", False):
            self._teacher_hook_logged = True
            logger.warning(
                "native OPD: student_prompt=%d teacher_prompt=%d delta=%+d "
                "teacher_score(first response token)=%.3f",
                len(student_prompt),
                len(teacher_prompt),
                len(teacher_prompt) - len(student_prompt),
                scores[len(student_prompt) - 1][0],
            )


class TeacherTemplateAgentLoopManager(AgentLoopManager):
    """Install the worker override without patching veRL source files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(TeacherTemplateAgentLoopWorker)
