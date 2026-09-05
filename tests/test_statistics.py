from omniopd.statistics import paired_hierarchical_bootstrap


def test_paired_bootstrap_recovers_constant_difference():
    treatment = {1: {"a": 1, "b": 0}, 2: {"a": 1, "b": 0}}
    control = {1: {"a": 0, "b": 0}, 2: {"a": 0, "b": 0}}
    result = paired_hierarchical_bootstrap(treatment, control, replicates=2000, rng_seed=1)
    assert result.estimate == 0.5
    assert result.lower <= result.estimate <= result.upper
    assert result.resampled_seeds


def test_crossed_bootstrap_keeps_the_same_sampled_games_across_training_seeds():
    # Treatment-control differences vary by game but are identical across
    # seeds. A crossed bootstrap must preserve that common game effect.
    treatment = {
        1: {"easy": 1, "hard": 0},
        2: {"easy": 1, "hard": 0},
    }
    control = {
        1: {"easy": 0, "hard": 0},
        2: {"easy": 0, "hard": 0},
    }
    result = paired_hierarchical_bootstrap(
        treatment, control, replicates=1000, rng_seed=3, resample_seeds=False
    )
    assert result.lower == 0.0
    assert result.upper == 1.0
