from __future__ import annotations

from typing import Any


def weighted_causal_ce_components(
    logits: Any,
    input_ids: Any,
    target_token_mask: Any,
    state_weights: Any,
) -> tuple[Any, Any]:
    """State/token-normalized causal CE with an explicit alignment contract.

    target_token_mask[b, t] marks whether input_ids[b, t] is a supervised
    target. Because logits[:, t-1] predicts input_ids[:, t], the aligned mask is
    target_token_mask[:, 1:], never target_token_mask[:, :-1].
    """

    import torch
    import torch.nn.functional as F

    if logits.ndim != 3 or input_ids.ndim != 2 or target_token_mask.ndim != 2:
        raise ValueError("expected logits[B,T,V], input_ids[B,T], mask[B,T]")
    if logits.shape[:2] != input_ids.shape or input_ids.shape != target_token_mask.shape:
        raise ValueError("logits, input_ids and target_token_mask shapes are inconsistent")
    if state_weights.ndim != 1 or state_weights.shape[0] != input_ids.shape[0]:
        raise ValueError("state_weights must have shape [B]")

    token_loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]).float(),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(input_ids.shape[0], -1)
    aligned_mask = target_token_mask[:, 1:].to(token_loss.dtype)
    lengths = aligned_mask.sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("every sample must retain at least one target token")
    per_state = (token_loss * aligned_mask).sum(dim=1) / lengths
    weights = state_weights.to(per_state.dtype)
    denominator = weights.sum()
    return (per_state * weights).sum(), denominator


def weighted_causal_ce(
    logits: Any,
    input_ids: Any,
    target_token_mask: Any,
    state_weights: Any,
) -> Any:
    numerator, denominator = weighted_causal_ce_components(
        logits, input_ids, target_token_mask, state_weights
    )
    if denominator <= 0:
        raise ValueError("sum of state weights must be positive")
    return numerator / denominator


def weighted_microbatch_loss(
    logits: Any,
    input_ids: Any,
    target_token_mask: Any,
    state_weights: Any,
    *,
    optimizer_batch_normalizer: Any,
    data_parallel_size: int = 1,
) -> Any:
    """Return one additive microbatch contribution to an optimizer-batch loss.

    optimizer_batch_normalizer must be computed once before splitting the
    optimizer batch. For uniformly sampled rows, use the expected sampled mass
    ``r * dataset_total_weight / dataset_size`` rather than the random in-batch
    weight sum. Do not divide the returned value after backward.
    """

    numerator, _ = weighted_causal_ce_components(
        logits, input_ids, target_token_mask, state_weights
    )
    if optimizer_batch_normalizer <= 0:
        raise ValueError("optimizer_batch_normalizer must be positive")
    return numerator / optimizer_batch_normalizer * data_parallel_size


def mean_target_log_probability(
    model: Any,
    input_ids: Any,
    target_token_mask: Any,
    *,
    attention_mask: Any | None = None,
    position_ids: Any | None = None,
) -> Any:
    """Use the same token positions as training for M3 source/transfer scores."""

    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        labels = input_ids[:, 1:]
        selected = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        mask = target_token_mask[:, 1:].to(selected.dtype)
        lengths = mask.sum(dim=1)
        if torch.any(lengths <= 0):
            raise ValueError("every sample must retain target tokens")
        return (selected * mask).sum(dim=1) / lengths
