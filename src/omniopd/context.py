from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import Message
from .tokenization import apply_chat_template_ids


@dataclass(frozen=True)
class TruncationInfo:
    token_count: int
    dropped_pairs: int
    kept_pairs: int
    budget: int


class TaskPreservingTruncator:
    """Keep System+initial Task/Observation and the most recent complete pairs."""

    def __init__(
        self,
        tokenizer: Any,
        max_context_tokens: int = 4096,
        reserve_tokens: int = 256,
        *,
        enable_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_context_tokens = int(max_context_tokens)
        self.reserve_tokens = int(reserve_tokens)
        self.enable_thinking = bool(enable_thinking)
        if self.reserve_tokens < 0 or self.max_context_tokens <= self.reserve_tokens:
            raise ValueError(
                "reserve_tokens must be non-negative and smaller than max_context_tokens"
            )

    def _length(self, messages: list[Message]) -> int:
        ids = apply_chat_template_ids(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        return len(ids)

    @staticmethod
    def _validate(history: list[Message]) -> list[list[Message]]:
        roles = [message.get("role") for message in history]
        if len(history) < 2 or roles[:2] != ["system", "user"]:
            raise ValueError("history must begin with system,user")
        tail = history[2:]
        if len(tail) % 2:
            raise ValueError("query history must contain complete assistant,user pairs")
        pairs: list[list[Message]] = []
        for offset in range(0, len(tail), 2):
            pair = tail[offset : offset + 2]
            if [message.get("role") for message in pair] != ["assistant", "user"]:
                raise ValueError("history tail must alternate assistant,user")
            pairs.append([dict(message) for message in pair])
        return pairs

    def truncate(self, history: list[Message]) -> tuple[list[Message], TruncationInfo]:
        pairs = self._validate(history)
        base = [dict(history[0]), dict(history[1])]
        budget = self.max_context_tokens - self.reserve_tokens
        if self._length(base) > budget:
            raise ValueError("system and initial task/observation exceed the context budget")
        kept: list[list[Message]] = []
        for reverse_index, pair in enumerate(reversed(pairs)):
            candidate_pairs = [pair] + kept
            candidate = base + [message for item in candidate_pairs for message in item]
            if self._length(candidate) > budget:
                if reverse_index == 0:
                    # Returning `base` here would silently replace O_t with the
                    # initial observation. The current state is mandatory; a
                    # too-large current pair must fail closed instead.
                    raise ValueError(
                        "system, task anchor, and current action/observation pair "
                        "exceed the context budget"
                    )
                break
            kept = candidate_pairs
        result = base + [message for pair in kept for message in pair]
        return result, TruncationInfo(self._length(result), len(pairs) - len(kept), len(kept), budget)
