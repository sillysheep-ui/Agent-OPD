"""veRL v0.8.0 pilot hooks for distinct-prompt, sampled-token OPD.

This module is intentionally fail-closed: malformed Student outputs stop the
pilot instead of being silently re-sampled or relabeled. An invalid-rollout
budget protocol is required before confirmatory training.
"""

from __future__ import annotations

import os

from collections.abc import Mapping
from typing import Any

import ray
import torch
from transformers import AutoTokenizer
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from .opd_adapter import (
    align_teacher_rows_to_student_layout,
    audit_shared_token_id_space,
    describe_supervision_span,
    extract_strict_action_tokens,
    remap_teacher_scores_to_student_layout,
    response_mask_for_action_span,
)
from .prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT, replace_system
from .tokenization import apply_chat_template_ids


def _extra_info(sample_kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    value = sample_kwargs.get("extra_info")
    if not isinstance(value, Mapping) and hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, Mapping):
        raise ValueError("OPD rollout requires extra_info with Teacher prompt and action set")
    return value


class OmniOPDActionLoop(SingleTurnAgentLoop):
    """Student action rollout with an action-content-only response mask."""

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        if self.apply_chat_template_kwargs.get("enable_thinking") is not False:
            raise ValueError("OPD Student rollout requires explicit enable_thinking=False")
        output = await super().run(sampling_params, **kwargs)
        extra = _extra_info(kwargs)
        actions = extra.get("admissible_actions")
        if not isinstance(actions, (list, tuple)) or not actions:
            raise ValueError("OPD state has no admissible action set")
        require_admissible = (
            os.environ.get("OMNIOPD_REQUIRE_ADMISSIBLE_ACTION", "1") != "0"
        )
        content_ids, response_text, canonical_action = extract_strict_action_tokens(
            self.tokenizer,
            output.response_ids,
            actions,
            require_admissible=require_admissible,
        )
        # veRL fixes every response at the batch width; supervise the action
        # span in place instead of resizing the response.
        output.response_mask = response_mask_for_action_span(
            len(output.response_ids), len(content_ids)
        )
        # veRL's rollout/reward pipeline requires rm_scores even when OPD
        # deliberately disables task rewards. This zero is a plumbing value,
        # not an ALFWorld success signal.
        output.reward_score = 0.0
        output.extra_fields["opd_canonical_action"] = canonical_action
        output.extra_fields["opd_scored_action_tokens"] = len(content_ids)
        # Ledger value: does this turn carry an action line, the whole emitted
        # response, or no content at all? Recorded instead of raised so one
        # degenerate turn cannot abort the run.
        output.extra_fields["opd_supervision"] = describe_supervision_span(
            content_ids, response_text
        )
        return output


class OmniOPDAgentLoopWorker(AgentLoopWorker):
    """Score Student action IDs under Teacher P_T, then remap to Student layout."""

    async def _compute_teacher_logprobs(
        self,
        output,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> None:
        require_admissible = (
            os.environ.get("OMNIOPD_REQUIRE_ADMISSIBLE_ACTION", "1") != "0"
        )
        if validate or not self.distillation_enabled:
            return
        if sample_kwargs is None:
            raise ValueError("OPD Teacher scoring requires the dataset sample metadata")
        if len(self.teacher_server_manager.teacher_model_configs) != 1:
            raise ValueError("pilot OPD scorer supports exactly one frozen Teacher")
        extra = _extra_info(sample_kwargs)
        student_messages = [dict(message) for message in sample_kwargs["raw_prompt"]]
        if (
            not student_messages
            or student_messages[0].get("role") != "system"
            or not str(student_messages[0].get("content", "")).strip()
            or student_messages[-1].get("role") != "user"
        ):
            raise ValueError(
                "OPD Student prompt must be a system message followed by the state history"
            )
        teacher_messages = extra.get("teacher_prompt")
        # Two contracts are legitimate: the paper's separate P_T view, and
        # standard token OPD where the Teacher scores the Student prefix in the
        # identical context. Both require the non-system history to be the same.
        if (
            not teacher_messages
            or [dict(message) for message in teacher_messages][1:]
            != [dict(message) for message in student_messages][1:]
        ):
            raise ValueError("Teacher prompt does not share the Student state history")
        actions = extra.get("admissible_actions")
        content_ids, response_text, _ = extract_strict_action_tokens(
            self.tokenizer,
            response_ids,
            actions,
            require_admissible=require_admissible,
        )
        output.extra_fields["opd_supervision"] = describe_supervision_span(
            content_ids, response_text
        )
        expected_student_ids = apply_chat_template_ids(
            self.tokenizer,
            student_messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if expected_student_ids != prompt_ids:
            raise ValueError("veRL altered the Student prompt before OPD Teacher scoring")

        teacher_key = next(iter(self.teacher_server_manager.teacher_model_configs))
        teacher_config = self.teacher_server_manager.teacher_model_configs[teacher_key]
        if not hasattr(self, "_omniopd_teacher_tokenizer"):
            self._omniopd_teacher_tokenizer = AutoTokenizer.from_pretrained(
                teacher_config.model_path,
                use_fast=True,
                local_files_only=True,
                trust_remote_code=False,
            )
            audit_shared_token_id_space(self.tokenizer, self._omniopd_teacher_tokenizer)
        teacher_prompt_ids = apply_chat_template_ids(
            self._omniopd_teacher_tokenizer,
            teacher_messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        max_model_len = teacher_config.inference.max_model_len
        if max_model_len is not None and len(teacher_prompt_ids) + len(content_ids) + 1 > max_model_len:
            raise ValueError("Teacher prompt plus Student action exceeds Teacher context limit")
        teacher_ids, teacher_logprobs = (
            await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=teacher_prompt_ids + content_ids,
                routing_key=teacher_key,
            )
        )
        remapped_ids, remapped_scores = remap_teacher_scores_to_student_layout(
            student_prompt_ids=prompt_ids,
            teacher_prompt_ids=teacher_prompt_ids,
            student_response_ids=content_ids,
            teacher_scored_ids=teacher_ids.tolist(),
            teacher_scored_logprobs=teacher_logprobs.tolist(),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Align the Teacher rows with the Student sequence width veRL fixes for
        # the batch; the action span alone can be shorter than the response.
        remapped_ids, remapped_scores = align_teacher_rows_to_student_layout(
            student_prompt_length=len(prompt_ids),
            student_response_ids=response_ids,
            teacher_rows=remapped_scores,
            teacher_ids=remapped_ids,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output.extra_fields["teacher_ids"] = torch.tensor(remapped_ids, dtype=torch.int32)
        output.extra_fields["teacher_logprobs"] = torch.tensor(
            remapped_scores, dtype=torch.float32
        )
        output.extra_fields["opd_teacher_prompt_tokens"] = len(teacher_prompt_ids)
        output.extra_fields["opd_teacher_scored_tokens"] = len(content_ids)


class OmniOPDAgentLoopManager(AgentLoopManager):
    """Install the repository-owned worker without patching veRL source files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(OmniOPDAgentLoopWorker)
