"""Fail-closed preparation and alignment contracts for token-level OPD.

These helpers prepare *Student* prompts from immutable ALFWorld states and
preserve a separate *Teacher* prompt for scoring Student-generated tokens.
They do not themselves launch veRL or claim to implement an online environment
loop. In particular, no Teacher-generated action is used as a target here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT, replace_system
from .parser import parse_action
from .schema import RolloutTurn


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompts_for_opd_turn(turn: RolloutTurn) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return the distinct Student-generation and Teacher-scoring prompt views."""
    state = turn.state
    if state.state_source != "student":
        raise ValueError("fixed-pool token OPD requires Student-source states")
    messages = [dict(message) for message in state.messages]
    if not messages or messages[0] != {"role": "system", "content": STUDENT_SYSTEM_PROMPT}:
        raise ValueError("state must contain the canonical Student system prompt")
    expected_roles = ["system"] + [
        "user" if index % 2 else "assistant" for index in range(1, len(messages))
    ]
    if len(messages) < 2 or [message.get("role") for message in messages] != expected_roles:
        raise ValueError("state messages must end in a user turn with alternating history")
    if messages[-1]["role"] != "user":
        raise ValueError("state prompt must stop before the Student action")
    if any(not isinstance(message.get("content"), str) for message in messages):
        raise ValueError("state messages must contain text content")
    return messages, replace_system(messages, TEACHER_SYSTEM_PROMPT)


def build_fixed_pool_opd_prompts(
    turns: Iterable[RolloutTurn], selections: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Create veRL RLHF prompt rows without leaking behavior/Teacher actions.

    `extra_info.teacher_prompt` is reserved for a repository-owned Teacher
    scorer. Upstream veRL scores the Student prompt verbatim and must not be
    presented as implementing the paper's distinct P_T contract.
    """

    turns = list(turns)
    by_hash = {turn.state.state_hash: turn for turn in turns}
    if not turns or len(by_hash) != len(turns):
        raise ValueError("state pool must be non-empty with unique state hashes")
    selections = list(selections)
    selected_hashes = [str(row["state_hash"]) for row in selections]
    if not selections or len(set(selected_hashes)) != len(selections):
        raise ValueError("selection must be non-empty with unique state hashes")
    if any(state_hash not in by_hash for state_hash in selected_hashes):
        raise ValueError("selection contains a state absent from the frozen pool")
    counts_by_game = Counter(by_hash[state_hash].state.game_id for state_hash in selected_hashes)
    rows = []
    for index, (state_hash, selection) in enumerate(zip(selected_hashes, selections)):
        turn = by_hash[state_hash]
        state = turn.state
        if (
            selection.get("protocol_version") != "omniopd-v1"
            or str(selection.get("game_id")) != state.game_id
            or isinstance(selection.get("turn_index"), bool)
            or selection.get("turn_index") != state.turn_index
        ):
            raise ValueError("selection row does not identify its frozen state")
        student_prompt, teacher_prompt = prompts_for_opd_turn(turn)
        weight = 1.0 / counts_by_game[state.game_id]
        rows.append(
            {
                "data_source": "alfworld_fixed_state_token_opd",
                "prompt": student_prompt,
                "extra_info": {
                    "index": index,
                    "state_hash": state_hash,
                    "game_id": state.game_id,
                    "turn_index": state.turn_index,
                    "admissible_actions": list(state.admissible_actions),
                    "selection_policy": selection.get("selection_policy"),
                    "state_weight": weight,
                    "teacher_prompt": teacher_prompt,
                },
            }
        )
    return rows


def audit_shared_token_id_space(student_tokenizer: Any, teacher_tokenizer: Any) -> dict[str, Any]:
    """Require identical ID→token meaning before reusing Student IDs for Teacher scoring.

    This is stronger than checking model families or vocabulary lengths. Prompt
    templates are deliberately audited separately because P_S and P_T differ.
    """

    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    if not student_vocab or student_vocab != teacher_vocab:
        raise ValueError("Student and Teacher token-to-ID vocabularies differ")
    if len(set(student_vocab.values())) != len(student_vocab):
        raise ValueError("tokenizer vocabulary contains duplicate IDs")
    special_names = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    student_special = {name: getattr(student_tokenizer, name, None) for name in special_names}
    teacher_special = {name: getattr(teacher_tokenizer, name, None) for name in special_names}
    if student_special != teacher_special:
        raise ValueError("Student and Teacher special-token IDs differ")
    if sorted(student_tokenizer.all_special_ids) != sorted(teacher_tokenizer.all_special_ids):
        raise ValueError("Student and Teacher special-token sets differ")
    student_template = getattr(student_tokenizer, "chat_template", None)
    teacher_template = getattr(teacher_tokenizer, "chat_template", None)
    if not isinstance(student_template, str) or not isinstance(teacher_template, str):
        raise ValueError("both Student and Teacher require explicit chat templates")
    student_backend = getattr(student_tokenizer, "backend_tokenizer", None)
    teacher_backend = getattr(teacher_tokenizer, "backend_tokenizer", None)
    if student_backend is None or teacher_backend is None:
        raise ValueError("fast tokenizers are required to audit decoding semantics")
    student_rules = json.loads(student_backend.to_str())
    teacher_rules = json.loads(teacher_backend.to_str())
    for rules in (student_rules, teacher_rules):
        rules.pop("truncation", None)
        rules.pop("padding", None)
    if student_rules != teacher_rules:
        raise ValueError("Student and Teacher tokenizer backend rules differ")
    return {
        "vocab_size": len(student_vocab),
        "vocab_sha256": _canonical_sha256(student_vocab),
        "backend_sha256": _canonical_sha256(student_rules),
        "student_chat_template_sha256": hashlib.sha256(
            student_template.encode("utf-8")
        ).hexdigest(),
        "teacher_chat_template_sha256": hashlib.sha256(
            teacher_template.encode("utf-8")
        ).hexdigest(),
        "chat_templates_equal": student_template == teacher_template,
        "special_token_ids": student_special,
    }


def align_teacher_sampled_token_logprobs(
    *,
    teacher_prompt_ids: Sequence[int],
    student_response_ids: Sequence[int],
    scored_sequence_ids: Sequence[int],
    scored_token_logprobs: Sequence[float | None],
) -> list[float]:
    """Slice Teacher prompt logprobs at Student response positions, not prompt positions.

    `scored_token_logprobs[j]` must mean log p(token_j | tokens_<j), as in
    vLLM prompt_logprobs. The first sequence token may have no logprob; all
    response tokens must have finite scores. This helper is for sampled-token
    losses (k1/k3), not top-k distributional losses.
    """

    prompt = [int(token) for token in teacher_prompt_ids]
    response = [int(token) for token in student_response_ids]
    scored = [int(token) for token in scored_sequence_ids]
    if not prompt or not response or scored != prompt + response:
        raise ValueError("Teacher must score exactly its prompt plus the Student response")
    if len(scored_token_logprobs) != len(scored):
        raise ValueError("Teacher token scores do not align with scored token IDs")
    result = list(scored_token_logprobs[len(prompt) :])
    if any(value is None or not math.isfinite(float(value)) for value in result):
        raise ValueError("Student response has missing or non-finite Teacher logprobs")
    return [float(value) for value in result]


def remap_teacher_scores_to_student_layout(
    *,
    student_prompt_ids: Sequence[int],
    teacher_prompt_ids: Sequence[int],
    student_response_ids: Sequence[int],
    teacher_scored_ids: Sequence[Sequence[int]],
    teacher_scored_logprobs: Sequence[Sequence[float]],
    pad_token_id: int,
) -> tuple[list[list[int]], list[list[float]]]:
    """Align veRL *left-shifted* Teacher scores to the Student sequence width.

    veRL's `extract_prompt_logprobs` stores the score/ID of token i+1 at
    sequence position i and appends a dummy final row. Later,
    `no_padding_2_padding` slices from the *last prompt position* so that the
    score for the first response token is read at that position. Preserve
    this convention even when Teacher and Student prompt lengths differ.
    This adapter rejects top-k outputs and is not valid for forward_kl_topk.
    """

    student_prompt = [int(value) for value in student_prompt_ids]
    teacher_prompt = [int(value) for value in teacher_prompt_ids]
    response = [int(value) for value in student_response_ids]
    if not student_prompt or not teacher_prompt or not response:
        raise ValueError("Student prompt, Teacher prompt, and response must be non-empty")
    expected = teacher_prompt + response
    if len(teacher_scored_ids) != len(expected) or len(teacher_scored_logprobs) != len(expected):
        raise ValueError("Teacher score length does not match its prompt plus Student response")
    if any(len(row) != 1 for row in teacher_scored_ids) or any(
        len(row) != 1 for row in teacher_scored_logprobs
    ):
        raise ValueError("sampled-token OPD requires one Teacher score per position")
    actual = [int(row[0]) for row in teacher_scored_ids]
    if actual != expected[1:] + [0]:
        raise ValueError("veRL-shifted Teacher token IDs disagree with the requested sequence")
    try:
        response_scores = [
            float(row[0])
            for row in teacher_scored_logprobs[
                len(teacher_prompt) - 1 : len(teacher_prompt) - 1 + len(response)
            ]
        ]
    except (TypeError, ValueError) as error:
        raise ValueError("Teacher response has a missing or invalid token score") from error
    if any(not math.isfinite(score) for score in response_scores):
        raise ValueError("Teacher response contains missing or non-finite token scores")
    remapped_ids = (
        [[int(pad_token_id)] for _ in student_prompt[:-1]]
        + [[token] for token in response]
        + [[int(pad_token_id)]]
    )
    remapped_scores = (
        [[0.0] for _ in student_prompt[:-1]]
        + [[score] for score in response_scores]
        + [[0.0]]
    )
    return remapped_ids, remapped_scores


def extract_strict_action_tokens(tokenizer: Any, generated_ids: Sequence[int], admissible_actions):
    """Require one executable action line ending in EOS, with EOS masked out.

    This fail-closed pilot policy is not yet a general invalid-rollout strategy
    for confirmatory OPD training.
    """

    ids = [int(token_id) for token_id in generated_ids]
    if len(ids) < 2 or ids[-1] != tokenizer.eos_token_id:
        raise ValueError("Student did not finish an action with its EOS token")
    content_ids = ids[:-1]
    special_ids = set(tokenizer.all_special_ids)
    if any(token_id in special_ids for token_id in content_ids):
        raise ValueError("Student action contains a special/reasoning token")
    response_text = tokenizer.decode(
        content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if not re.fullmatch(r"Action:[^\r\n]+(?:\r?\n)?", response_text):
        raise ValueError(f"Student response is not exactly one action line: {response_text!r}")
    parsed = parse_action(response_text, admissible_actions)
    if not parsed.valid:
        raise ValueError(
            f"Student action is not admissible: {parsed.failure_reason}: {response_text!r}"
        )
    return content_ids, response_text, parsed.canonical_action
