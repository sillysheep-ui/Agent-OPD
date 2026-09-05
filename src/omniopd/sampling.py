from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Literal, TypeVar

T = TypeVar("T")

ThinkingMode = Literal["enabled", "disabled", "provider_default"]
ReasoningEffort = Literal["low", "high", "max"]


def stable_shuffled(values: Iterable[T], *, seed: int) -> list[T]:
    """Shuffle from a sorted base so upstream filesystem order cannot leak in."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    result = sorted(values)
    random.Random(seed).shuffle(result)
    return result


@dataclass(frozen=True)
class SamplingProtocol:
    """Explicit provider decoding contract for one policy role."""

    thinking_mode: ThinkingMode
    temperature: float | None
    reasoning_effort: ReasoningEffort | None = None

    def validate(self, *, samples_per_state: int = 1) -> None:
        if samples_per_state <= 0:
            raise ValueError("samples_per_state must be positive")
        if self.thinking_mode == "enabled":
            if self.temperature is not None:
                raise ValueError("thinking mode requires temperature=None")
            if self.reasoning_effort not in {"low", "high", "max"}:
                raise ValueError("thinking mode requires reasoning_effort=low/high/max")
            return
        if self.thinking_mode == "provider_default":
            if self.reasoning_effort is not None:
                raise ValueError("provider-default mode cannot claim a reasoning effort")
            if self.temperature is None or not 0.0 <= self.temperature <= 2.0:
                raise ValueError("provider-default mode requires an explicit temperature in [0,2]")
            return
        if self.reasoning_effort is not None:
            raise ValueError("non-thinking mode cannot set reasoning_effort")
        if self.temperature is None or not 0.0 <= self.temperature <= 2.0:
            raise ValueError("non-thinking mode requires an explicit temperature in [0,2]")
        if samples_per_state > 1 and self.temperature == 0.0:
            raise ValueError(
                "N>1 with temperature=0 is a degenerate depth protocol; use positive "
                "temperature or an explicitly sampled thinking-mode policy"
            )

    def to_dict(self) -> dict[str, str | float | None]:
        return {
            "thinking_mode": self.thinking_mode,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }

    @property
    def policy_thinking_mode(self) -> Literal["enabled", "disabled"] | None:
        return None if self.thinking_mode == "provider_default" else self.thinking_mode
