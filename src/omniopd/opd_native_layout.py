"""Prompt rendering and score layout for the veRL-native OPD run.

Kept separate from ``opd_native`` so the two contracts that must be exactly
right can be checked without torch, Ray or a running Teacher:

* the Teacher prompt is rendered with the Teacher's own template, which makes it
  longer than the Student's by the empty thinking block, and
* the Teacher's scores land where veRL reads them, i.e. the score of response
  token ``i`` at sequence position ``student_prompt_length - 1 + i``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def teacher_messages(sample: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """The conversation to score: the same history the Student is acting in."""

    if sample is None:
        raise ValueError("Teacher scoring requires the dataset sample")
    messages = sample.get("raw_prompt")
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("dataset sample must expose a non-empty raw_prompt")
    history = [dict(message) for message in messages]
    if history[0].get("role") != "system" or history[-1].get("role") != "user":
        raise ValueError("state history must start with a system message and end with a user turn")
    if any(not isinstance(message.get("content"), str) for message in history):
        raise ValueError("state history must contain text content")
    return history


def render_prompt(tokenizer: Any, history: Sequence[Mapping[str, str]]) -> list[int]:
    """Render a conversation with a tokenizer's own template, thinking disabled."""

    encoded = tokenizer.apply_chat_template(
        [dict(message) for message in history],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    while isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def teacher_rows_for_student_layout(
    *,
    student_prompt: Sequence[int],
    teacher_prompt: Sequence[int],
    response: Sequence[int],
    teacher_ids: Sequence[Sequence[int]],
    teacher_logprobs: Sequence[Sequence[float]],
    pad_token_id: int,
) -> tuple[list[list[int]], list[list[float]]]:
    """Place Teacher scores at the positions veRL reads them from.

    veRL stores the score of token ``i+1`` at sequence position ``i`` (and a
    dummy final row), then reads a response token's score from the position
    ``student_prompt_length - 1 + i``. The Teacher's own prompt can be longer
    than the Student's, so the Teacher's slice starts at its own last prompt
    position. Every non-response position is filled with a value the response
    mask discards.
    """

    student_prompt = [int(token) for token in student_prompt]
    teacher_prompt = [int(token) for token in teacher_prompt]
    response = [int(token) for token in response]
    if not student_prompt or not teacher_prompt or not response:
        raise ValueError("Student prompt, Teacher prompt and response must be non-empty")
    if len(teacher_ids) != len(teacher_prompt) + len(response):
        raise ValueError("Teacher rows must cover its prompt plus the response")
    if len(teacher_ids) != len(teacher_logprobs):
        raise ValueError("Teacher ids and scores must have the same length")
    if any(len(row) != 1 for row in teacher_logprobs):
        raise ValueError("this layout supports one Teacher score per position")
    start = len(teacher_prompt) - 1
    scores = [[float(row[0])] for row in teacher_logprobs[start : start + len(response)]]
    if len(scores) != len(response):
        raise ValueError("Teacher does not cover every response token")
    ids = [[int(pad_token_id)] for _ in range(len(student_prompt) - 1)]
    ids += [[token] for token in response]
    ids += [[int(pad_token_id)]]
    out_scores = [[0.0] for _ in range(len(student_prompt) - 1)]
    out_scores += scores
    out_scores += [[0.0]]
    return ids, out_scores
