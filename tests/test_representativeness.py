from omniopd.representativeness import (
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
