from omniopd.sampler import ZeroPaddedDistributedSampler


def test_distributed_padding_is_present_but_has_zero_objective_weight():
    dataset = list(range(5))
    shards = [
        list(
            ZeroPaddedDistributedSampler(
                dataset, num_replicas=2, rank=rank, shuffle=False
            )
        )
        for rank in range(2)
    ]
    assert [len(shard) for shard in shards] == [3, 3]
    combined = [item for shard in shards for item in shard]
    assert sum(weight for _, weight in combined) == 5.0
    assert sum(weight == 0.0 for _, weight in combined) == 1
