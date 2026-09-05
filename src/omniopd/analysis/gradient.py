from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class GradientAlignment:
    dot_product: float
    cosine: float
    correction_norm: float
    reference_norm: float


def gradient_alignment(correction: Any, reference: Any) -> GradientAlignment:
    import torch

    correction = correction.float().reshape(-1)
    reference = reference.float().reshape(-1)
    if correction.shape != reference.shape:
        raise ValueError("gradient vectors have different shapes")
    dot = torch.dot(correction, reference)
    correction_norm = torch.linalg.vector_norm(correction)
    reference_norm = torch.linalg.vector_norm(reference)
    cosine = dot / (correction_norm * reference_norm).clamp_min(1e-12)
    return GradientAlignment(
        float(dot), float(cosine), float(correction_norm), float(reference_norm)
    )


def online_mean(vectors: Iterable[Any]) -> Any:
    """Avoid retaining every high-dimensional LoRA gradient in memory."""

    mean = None
    count = 0
    for vector in vectors:
        value = vector.detach().float()
        count += 1
        mean = value.clone() if mean is None else mean + (value - mean) / count
    if mean is None:
        raise ValueError("at least one gradient vector is required")
    return mean


def assert_held_out_reference_games(
    correction_games: Iterable[str], reference_games: Iterable[str]
) -> None:
    """Prevent a selection-defined complement from masquerading as G_*."""

    correction = set(correction_games)
    reference = set(reference_games)
    if not reference:
        raise ValueError("the reference-gradient game set cannot be empty")
    overlap = correction & reference
    if overlap:
        raise ValueError(
            f"reference gradients must use preregistered held-out games; overlap={sorted(overlap)[:5]}"
        )
