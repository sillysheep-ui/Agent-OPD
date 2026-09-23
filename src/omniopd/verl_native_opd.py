"""The only code this project adds to veRL's native on-policy distillation path.

veRL's stock worker scores the Teacher on the Student's own rendered ids
(`verl/experimental/agent_loop/agent_loop.py: _compute_teacher_logprobs` passes
`prompt_ids + response_ids`). That contract is correct whenever Student and
Teacher share a chat template. Our pair does not: the Student is an
Instruct-2507 release whose template has no thinking branch, while the Teacher
is a hybrid-thinking release whose `enable_thinking=False` template inserts an
empty thinking block. Measured consequence: with the Student's rendering the
Teacher's argmax at the first action token is `<think>` and it assigns the token
the Student actually produced about 21 nats less probability, in 100% of the
turns we sampled.

This module therefore overrides exactly one method: the Teacher is asked about
the same conversation rendered with *its own* tokenizer and template, and the
returned scores are mapped back onto the Student's layout. Rollout, response
mask, normalisation, loss, optimiser and checkpointing all stay stock, so the
arm remains an instance of the framework's own OPD.

The sampled-token estimators (k1/k2/k3) emit one score per position; the
distributional mode (`forward_kl_topk`) emits K columns and is rejected here on
purpose until the remap helpers grow a K-column path.
"""

from __future__ import annotations

import logging
from typing import Any

import ray
import torch
from transformers import AutoTokenizer
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker

from .opd_adapter import (
    align_teacher_rows_to_student_layout,
    audit_shared_token_id_space,
    remap_teacher_scores_to_student_layout,
    render_teacher_prompt_ids,
    teacher_state_messages,
)

logger = logging.getLogger(__name__)


class TeacherTemplateAgentLoopWorker(AgentLoopWorker):
    """Stock worker with the Teacher prompt rendered by the Teacher's template."""

    def _teacher_tokenizer(self):
        configs = self.teacher_server_manager.teacher_model_configs
        if len(configs) != 1:
            raise ValueError("this hook supports exactly one frozen Teacher")
        if not hasattr(self, "_omniopd_teacher_tokenizer"):
            key = next(iter(configs))
            tokenizer = AutoTokenizer.from_pretrained(
                configs[key].model_path,
                use_fast=True,
                local_files_only=True,
                trust_remote_code=False,
            )
            audit_shared_token_id_space(self.tokenizer, tokenizer)
            self._omniopd_teacher_tokenizer = tokenizer
        return self._omniopd_teacher_tokenizer

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
        student_prompt_ids = [int(token) for token in prompt_ids]
        student_response_ids = [int(token) for token in response_ids]
        if not student_response_ids:
            raise ValueError("native OPD needs a non-empty Student response")

        teacher_tokenizer = self._teacher_tokenizer()
        teacher_prompt_ids = render_teacher_prompt_ids(
            teacher_tokenizer, teacher_state_messages(sample_kwargs)
        )
        teacher_key = next(iter(self.teacher_server_manager.teacher_model_configs))
        teacher_ids, teacher_logprobs = (
            await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=teacher_prompt_ids + student_response_ids,
                routing_key=teacher_key,
            )
        )
        scored_ids = teacher_ids.tolist()
        scored_logprobs = teacher_logprobs.tolist()
        width = {len(row) for row in scored_ids} | {len(row) for row in scored_logprobs}
        if width != {1}:
            raise ValueError(
                "this hook supports one Teacher score per position; for the distributional "
                f"mode (forward_kl_topk) the remap helpers need a K-column path, got width {width}"
            )

        remapped_ids, remapped_scores = remap_teacher_scores_to_student_layout(
            student_prompt_ids=student_prompt_ids,
            teacher_prompt_ids=teacher_prompt_ids,
            student_response_ids=student_response_ids,
            teacher_scored_ids=scored_ids,
            teacher_scored_logprobs=scored_logprobs,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        remapped_ids, remapped_scores = align_teacher_rows_to_student_layout(
            student_prompt_length=len(student_prompt_ids),
            student_response_ids=student_response_ids,
            teacher_rows=remapped_scores,
            teacher_ids=remapped_ids,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output.extra_fields["teacher_ids"] = torch.tensor(remapped_ids, dtype=torch.int32)
        output.extra_fields["teacher_logprobs"] = torch.tensor(
            remapped_scores, dtype=torch.float32
        )
        output.extra_fields["opd_student_prompt_tokens"] = len(student_prompt_ids)
        output.extra_fields["opd_teacher_prompt_tokens"] = len(teacher_prompt_ids)
        output.extra_fields["opd_teacher_template_delta"] = (
            len(teacher_prompt_ids) - len(student_prompt_ids)
        )
        # One line per worker, so a pilot run can show whether the hook did what
        # it claims: how many tokens the Teacher template adds, and the Teacher's
        # score for the Student's first response token. Without the hook that
        # score sits near -21 on a thinking-capable Teacher; with it, near 0.
        if not getattr(self, "_omniopd_hook_logged", False):
            self._omniopd_hook_logged = True
            first = float(remapped_scores[len(student_prompt_ids) - 1][0])
            logger.warning(
                "native OPD hook: student_prompt=%d teacher_prompt=%d delta=%+d "
                "teacher_score(first response token)=%.3f",
                len(student_prompt_ids),
                len(teacher_prompt_ids),
                len(teacher_prompt_ids) - len(student_prompt_ids),
                first,
            )


class TeacherTemplateAgentLoopManager(AgentLoopManager):
    """Install the hook without patching veRL source files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(TeacherTemplateAgentLoopWorker)
