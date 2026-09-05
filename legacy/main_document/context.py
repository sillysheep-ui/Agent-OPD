from __future__ import annotations
 
from dataclasses import asdict, dataclass
from typing import Any
 
 
@dataclass(frozen=True)
class TruncationInfo:
    token_count: int
    max_context_tokens: int
    reserve_tokens: int
    dropped_pairs: int
    kept_pairs: int
    full_pairs: int
 
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
 
 
class TaskPreservingTruncator:
    """Preserve System + Task and add recent complete interaction pairs.
 
    Expected history shape at model-query time:
      [system, user(task),
       assistant(action_0), user(observation_1),
       assistant(action_1), user(observation_2), ...]
 
    The truncator never removes System/Task and never cuts an interaction pair
    in half. Token counts use the Student tokenizer's actual chat template.
    """
 
    def __init__(
        self,
        tokenizer,
        *,
        max_context_tokens: int = 4096,
        reserve_tokens: int = 256,
        enable_thinking: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_context_tokens = int(max_context_tokens)
        self.reserve_tokens = int(reserve_tokens)
        self.enable_thinking = bool(enable_thinking)
 
        if self.max_context_tokens <= self.reserve_tokens:
            raise ValueError("max_context_tokens must be larger than reserve_tokens")
 
    def _chat_token_len(self, messages: list[dict[str, str]]) -> int:
        kwargs = dict(
            tokenize=True,
            add_generation_prompt=True,
        )
        # Qwen-family templates may expose enable_thinking; older templates may not.
        try:
            ids = self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=self.enable_thinking,
                **kwargs,
            )
        except TypeError:
            ids = self.tokenizer.apply_chat_template(messages, **kwargs)
 
        # Some tokenizers return tensors only when return_tensors is requested;
        # by default this should be a Python list.
        try:
            return len(ids)
        except TypeError as exc:
            raise RuntimeError("Unexpected tokenizer.apply_chat_template output") from exc
 
    @staticmethod
    def _interaction_pairs(
        history: list[dict[str, str]],
    ) -> list[list[dict[str, str]]]:
        tail = history[2:]
        if not tail:
            return []
 
        if len(tail) % 2 != 0:
            raise ValueError(
                "Malformed model-query history: after System+Task, expected complete "
                "(assistant action, user observation) pairs."
            )
 
        pairs: list[list[dict[str, str]]] = []
        for i in range(0, len(tail), 2):
            a, o = tail[i], tail[i + 1]
            if a.get("role") != "assistant" or o.get("role") != "user":
                raise ValueError(
                    "Malformed model-query history roles: expected assistant,user pairs; "
                    f"got {a.get('role')!r},{o.get('role')!r} at tail indices {i},{i+1}."
                )
            pairs.append([a, o])
        return pairs
 
    def truncate(
        self,
        full_history: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], TruncationInfo]:
        if len(full_history) < 2:
            raise ValueError("History must include at least System and Task messages")
        if full_history[0].get("role") != "system" or full_history[1].get("role") != "user":
            raise ValueError("History must begin with [system, user(task)]")
 
        base = [dict(full_history[0]), dict(full_history[1])]
        pairs = self._interaction_pairs(full_history)
 
        base_len = self._chat_token_len(base)
        budget_limit = self.max_context_tokens - self.reserve_tokens
        if base_len > budget_limit:
            raise RuntimeError(
                "System+Task alone exceed the context budget: "
                f"{base_len} > {budget_limit}. Increase max_context_tokens or shorten prompts."
            )
 
        kept_reversed: list[list[dict[str, str]]] = []
        for pair in reversed(pairs):
            candidate_pairs = list(reversed(kept_reversed + [pair]))
            candidate = base + [m for p in candidate_pairs for m in p]
            if self._chat_token_len(candidate) <= budget_limit:
                kept_reversed.append(pair)
            else:
                break
 
        kept_pairs = list(reversed(kept_reversed))
        truncated = base + [dict(m) for p in kept_pairs for m in p]
        token_count = self._chat_token_len(truncated)
 
        info = TruncationInfo(
            token_count=token_count,
            max_context_tokens=self.max_context_tokens,
            reserve_tokens=self.reserve_tokens,
            dropped_pairs=len(pairs) - len(kept_pairs),
            kept_pairs=len(kept_pairs),
            full_pairs=len(pairs),
        )
        return truncated, info
