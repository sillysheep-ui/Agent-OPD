def test_weighted_loss_is_microbatch_partition_invariant():
    try:
        import torch
    except ModuleNotFoundError:
        from unittest import SkipTest

        raise SkipTest("optional torch dependency is not installed")

    from omniopd.loss import weighted_causal_ce, weighted_microbatch_loss

    torch.manual_seed(0)
    logits = torch.randn(3, 5, 11, requires_grad=True)
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5], [1, 3, 2, 7, 8], [1, 4, 6, 5, 2]], dtype=torch.long
    )
    masks = torch.tensor(
        [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 1]], dtype=torch.long
    )
    weights = torch.tensor([0.25, 0.25, 0.5])
    full = weighted_causal_ce(logits, input_ids, masks, weights)
    denominator = weights.sum()
    split = sum(
        weighted_microbatch_loss(
            logits[index : index + 1],
            input_ids[index : index + 1],
            masks[index : index + 1],
            weights[index : index + 1],
            optimizer_batch_normalizer=denominator,
        )
        for index in range(3)
    )
    assert torch.allclose(full, split, atol=1e-7)


def test_mask_alignment_supervises_the_first_target_not_the_previous_token():
    try:
        import torch
    except ModuleNotFoundError:
        from unittest import SkipTest

        raise SkipTest("optional torch dependency is not installed")

    from omniopd.loss import weighted_causal_ce

    # The target tokens occupy input positions 2 and 3. Loss must therefore use
    # logits positions 1 and 2. Position 0 deliberately predicts a wrong label.
    input_ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.full((1, 4, 4), -20.0)
    logits[0, 0, 0] = 20.0
    logits[0, 1, 2] = 20.0
    logits[0, 2, 3] = 20.0
    mask = torch.tensor([[0, 0, 1, 1]])
    loss = weighted_causal_ce(logits, input_ids, mask, torch.tensor([1.0]))
    assert loss.item() < 1e-6


def test_zero_weight_sampler_padding_has_zero_microbatch_contribution():
    try:
        import torch
    except ModuleNotFoundError:
        from unittest import SkipTest

        raise SkipTest("optional torch dependency is not installed")

    from omniopd.loss import weighted_microbatch_loss

    logits = torch.randn(1, 4, 5, requires_grad=True)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    mask = torch.tensor([[0, 0, 1, 1]])
    value = weighted_microbatch_loss(
        logits,
        input_ids,
        mask,
        torch.tensor([0.0]),
        optimizer_batch_normalizer=torch.tensor(1.0),
    )
    assert value.item() == 0.0


def test_uniform_sampling_normalizer_preserves_weights_for_batch_size_one():
    try:
        import torch
    except ModuleNotFoundError:
        from unittest import SkipTest

        raise SkipTest("optional torch dependency is not installed")

    from omniopd.verl_adapter import uniform_sampling_normalizer

    # Dataset weights [0.1, 0.9] sum to one. A uniformly sampled singleton has
    # expected mass 1/2, not its own random weight. Consequently the two
    # possible weighted loss contributions average to the global target.
    normalizer = uniform_sampling_normalizer(
        torch.tensor([1.0]), dataset_total_weight=1.0, dataset_size=2
    )
    assert normalizer.item() == 0.5
    losses = torch.tensor([2.0, 10.0])
    weights = torch.tensor([0.1, 0.9])
    contributions = weights * losses / normalizer
    assert torch.allclose(contributions.mean(), (weights * losses).sum())
