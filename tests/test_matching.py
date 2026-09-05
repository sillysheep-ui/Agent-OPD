from omniopd.matching import exact_stratified_sample


def test_matching_preserves_preregistered_stratum_counts():
    candidates = [
        {"sample_id": f"a{i}", "task": "a", "depth_bin": 0} for i in range(4)
    ] + [{"sample_id": f"b{i}", "task": "b", "depth_bin": 1} for i in range(3)]
    targets = [
        {"task": "a", "depth_bin": 0},
        {"task": "a", "depth_bin": 0},
        {"task": "b", "depth_bin": 1},
    ]
    selected = exact_stratified_sample(
        candidates, targets, strata=["task", "depth_bin"], rng_seed=3
    )
    assert sum(row["task"] == "a" for row in selected) == 2
    assert sum(row["task"] == "b" for row in selected) == 1
