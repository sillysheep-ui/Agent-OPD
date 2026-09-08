from omniopd.representativeness import (
    bootstrap_js,
    common_retained_game_support,
    game_balanced_probability_mass,
    paired_game_balanced_bootstrap_js,
    probability_mass,
)


def test_game_balanced_mass_does_not_let_long_games_dominate():
    clusters = [["a"] * 100, ["b"]]
    pooled = probability_mass(item for cluster in clusters for item in cluster)
    balanced = game_balanced_probability_mass(clusters)
    assert pooled["a"] > 0.99
    assert balanced == {"a": 0.5, "b": 0.5}


def test_paired_js_is_zero_for_equal_per_game_distributions():
    interval = paired_game_balanced_bootstrap_js(
        {"g1": ["a", "a"], "g2": ["b"]},
        {"g1": ["a"], "g2": ["b", "b"]},
        replicates=100,
        rng_seed=1,
    )
    assert interval.estimate == 0.0
    assert interval.lower == 0.0
    assert interval.upper == 0.0


def test_unpaired_js_uses_game_balanced_distributions():
    interval = bootstrap_js(
        [["a"] * 100, ["b"]],
        [["a"], ["b"] * 100],
        replicates=100,
        rng_seed=1,
    )
    assert interval.estimate == 0.0


def test_js_intervals_reject_single_game_cluster():
    try:
        bootstrap_js([["a"]], [["a"]], replicates=10)
    except ValueError as error:
        assert "two game clusters" in str(error)
    else:
        raise AssertionError("one game cannot identify game-cluster uncertainty")

    try:
        paired_game_balanced_bootstrap_js(
            {"g1": ["a"]}, {"g1": ["a"]}, replicates=10
        )
    except ValueError as error:
        assert "two paired games" in str(error)
    else:
        raise AssertionError("one paired game cannot identify game-cluster uncertainty")


def test_m2_uses_one_common_retained_game_support_across_groups():
    assert common_retained_game_support(
        {"A1": ["g1", "g2"], "A3": ["g2", "g3"]}
    ) == ("g2",)
    try:
        common_retained_game_support({"A1": ["g1"], "A3": ["g2"]})
    except ValueError as error:
        assert "incomparable" in str(error)
    else:
        raise AssertionError("disjoint M2 supports cannot be compared")
