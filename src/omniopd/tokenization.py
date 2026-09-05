from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

from .schema import Message


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    attention_mask: list[int]
    target_token_mask: list[int]
    target_length: int
    template_terminator_length: int


def _to_list(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("chat template mapping does not contain input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def apply_chat_template_ids(
    tokenizer: Any,
    messages: Sequence[Message],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> list[int]:
    """Normalize Transformers 4.x list and 5.x mapping return types."""

    return _to_list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    )


def encode_final_assistant_example(
    tokenizer: Any,
    messages: Sequence[Message],
    *,
    max_length: int | None = None,
    truncation: str = "error",
    enable_thinking: bool = False,
) -> EncodedExample:
    """Encode once under the real chat template and mark target token positions.

    The function fails closed when the generation prompt is not a prefix of the
    full conversation. It never falls back to concatenating independently
    tokenized strings, which can insert BOS tokens or omit assistant headers.
    """

    copied = [dict(message) for message in messages]
    if len(copied) < 3 or copied[-1].get("role") != "assistant":
        raise ValueError("messages must end with the supervised assistant response")
    if any(message.get("role") == "assistant" for message in copied[:-1]):
        # Final-turn training is intentional; history assistant actions remain context.
        pass
    prompt_messages = copied[:-1]
    prompt_ids = apply_chat_template_ids(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_ids = apply_chat_template_ids(
        tokenizer,
        copied,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    generation_bridge = ""
    if full_ids[: len(prompt_ids)] != prompt_ids:
        try:
            generation_bridge = _derive_generation_bridge(
                tokenizer,
                prompt_messages,
                enable_thinking=enable_thinking,
            )
        except Exception as error:
            raise ValueError(
                "chat template prefix mismatch and no safe generation bridge was derived"
            ) from error
        bridged = [*prompt_messages, dict(copied[-1])]
        bridged[-1]["content"] = generation_bridge + bridged[-1]["content"]
        full_ids = apply_chat_template_ids(
            tokenizer,
            bridged,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "chat template prefix mismatch even after deriving its generation bridge"
            )
    empty_completion = [
        *prompt_messages,
        {"role": "assistant", "content": generation_bridge},
    ]
    empty_ids = apply_chat_template_ids(
        tokenizer,
        empty_completion,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    if empty_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("empty assistant completion does not share the generation prefix")
    terminator = empty_ids[len(prompt_ids) :]
    if terminator and full_ids[-len(terminator) :] != terminator:
        raise ValueError("assistant template terminator is not stable across contents")
    mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    if not any(mask):
        raise ValueError("assistant target produced no trainable tokens")

    if max_length is not None and len(full_ids) > max_length:
        if truncation == "error":
            raise ValueError(f"encoded sequence length {len(full_ids)} exceeds {max_length}")
        if truncation != "left":
            raise ValueError("only left truncation is permitted for final-turn examples")
        drop = len(full_ids) - max_length
        if drop > len(prompt_ids):
            raise ValueError("truncation would remove part of the assistant target")
        full_ids = full_ids[drop:]
        mask = mask[drop:]
        if not any(mask):
            raise ValueError("truncation removed the complete target")
    return EncodedExample(
        full_ids,
        [1] * len(full_ids),
        mask,
        sum(mask),
        len(terminator),
    )


def encode_final_assistant_content(
    tokenizer: Any,
    messages: Sequence[Message],
    *,
    max_length: int | None = None,
    truncation: str = "error",
    enable_thinking: bool = False,
) -> EncodedExample:
    """Mark only assistant action-content tokens, excluding template syntax.

    Canonical training, M1, M3, and admissible-action uncertainty all use this
    mask, so the implemented token average is exactly the paper's action-token
    objective. Assistant headers, Qwen's generation-only thinking bridge, and
    turn terminators remain context with mask zero.
    """

    encoded = encode_final_assistant_example(
        tokenizer,
        messages,
        max_length=max_length,
        truncation=truncation,
        enable_thinking=enable_thinking,
    )
    prompt_length = len(encoded.input_ids) - encoded.target_length
    terminator_length = encoded.template_terminator_length
    content_end = len(encoded.input_ids) - terminator_length
    mask = [0] * len(encoded.input_ids)
    for index in range(prompt_length, content_end):
        mask[index] = 1
    if not any(mask):
        raise ValueError("assistant content produced no scoreable tokens")
    return EncodedExample(
        encoded.input_ids,
        encoded.attention_mask,
        mask,
        sum(mask),
        terminator_length,
    )


def _derive_generation_bridge(
    tokenizer: Any,
    prompt_messages: Sequence[Message],
    *,
    enable_thinking: bool,
) -> str:
    """Derive template text inserted only by ``add_generation_prompt``.

    Qwen3 non-thinking generation prompts include an empty thinking block,
    while a completed assistant message does not add it automatically. The
    bridge is inferred from the template itself and remains part of the masked
    prompt, never part of the action target.
    """

    common = {"tokenize": False, "enable_thinking": enable_thinking}
    base = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=False, **common
    )
    generation = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, **common
    )
    sentinel = "__OMNIOPD_ASSISTANT_CONTENT_SENTINEL_7D93A6__"
    completed = tokenizer.apply_chat_template(
        [*prompt_messages, {"role": "assistant", "content": sentinel}],
        add_generation_prompt=False,
        **common,
    )
    if not all(isinstance(value, str) for value in [base, generation, completed]):
        raise ValueError("tokenizer must return strings when tokenize=False")
    if not generation.startswith(base) or not completed.startswith(base):
        raise ValueError("chat template cannot be decomposed around the final assistant turn")
    generation_tail = generation[len(base) :]
    completed_tail = completed[len(base) :]
    if completed_tail.count(sentinel) != 1:
        raise ValueError("assistant content is transformed by the chat template")
    assistant_header = completed_tail.split(sentinel, 1)[0]
    if not assistant_header or not generation_tail.startswith(assistant_header):
        raise ValueError("generation and completed assistant headers do not match")
    bridge = generation_tail[len(assistant_header) :]
    if not bridge:
        raise ValueError("chat template mismatch has no derivable generation-only bridge")
    return bridge


def prediction_mask(target_token_mask: Sequence[int]) -> list[int]:
    """Align token-position masks with logits predicting input_ids[:, 1:]."""

    if len(target_token_mask) < 2:
        return []
    return [int(value) for value in target_token_mask[1:]]


def token_weights(target_token_mask: Sequence[int], state_weight: float) -> list[float]:
    if state_weight < 0:
        raise ValueError("state_weight must be non-negative")
    length = sum(int(value) for value in target_token_mask)
    if length <= 0:
        raise ValueError("target token mask is empty")
    return [float(value) * state_weight / length for value in target_token_mask]
