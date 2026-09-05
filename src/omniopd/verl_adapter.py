from __future__ import annotations

import math
from typing import Any, Iterable

from .loss import weighted_microbatch_loss


def uniform_sampling_normalizer(
    sampling_multiplier: Any,
    *,
    dataset_total_weight: float,
    dataset_size: int,
    distributed: bool = False,
) -> Any:
    """Return the fixed-mass normalizer for a uniform row sample.

    For ``r`` real rows sampled uniformly from a dataset of size ``D`` whose
    objective weights sum to ``W``, the expected sampled weight is ``rW/D``.
    Dividing the weighted numerator by that quantity is an unbiased stochastic
    estimator of ``sum_i w_i loss_i / W``. In contrast, dividing by the random
    in-batch sum of weights makes all weights disappear when the optimizer
    batch has one real row.
    """

    import torch

    if dataset_size <= 0 or not math.isfinite(dataset_total_weight) or dataset_total_weight <= 0:
        raise ValueError("dataset size and total objective weight must be positive")
    multipliers = sampling_multiplier.to(torch.float32)
    if multipliers.ndim != 1:
        raise ValueError("sampling_multiplier must have shape [B]")
    if not bool(torch.isfinite(multipliers).all()) or bool(
        ((multipliers != 0.0) & (multipliers != 1.0)).any()
    ):
        raise ValueError("sampling_multiplier must contain only finite 0/1 values")
    real_count = multipliers.sum()
    if distributed:
        torch.distributed.all_reduce(real_count, op=torch.distributed.ReduceOp.SUM)
    normalizer = real_count * (float(dataset_total_weight) / int(dataset_size))
    if not bool(torch.isfinite(normalizer)) or normalizer.item() <= 0:
        raise ValueError("optimizer batch has no real samples")
    return normalizer


def backward_optimizer_batch(
    *,
    model: Any,
    microbatches: Iterable[Any],
    normalizer: Any,
    data_parallel_size: int = 1,
) -> Any:
    """veRL-facing reference loop with microbatch-invariant weighting."""

    import torch

    logged_numerator = torch.zeros((), device=normalizer.device, dtype=torch.float32)
    for microbatch in microbatches:
        output = model(
            input_ids=microbatch["input_ids"],
            attention_mask=microbatch.get("attention_mask"),
            position_ids=microbatch.get("position_ids"),
            use_cache=False,
        )
        contribution = weighted_microbatch_loss(
            output.logits,
            microbatch["input_ids"],
            microbatch["target_token_mask"],
            microbatch["state_weight"],
            optimizer_batch_normalizer=normalizer,
            data_parallel_size=data_parallel_size,
        )
        contribution.backward()
        logged_numerator += contribution.detach().float()
    return logged_numerator
