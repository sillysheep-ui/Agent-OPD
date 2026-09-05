from omniopd.analysis.gradient import assert_held_out_reference_games


def test_m1_reference_games_must_be_held_out():
    assert_held_out_reference_games(["train-a"], ["probe-b"])
    try:
        assert_held_out_reference_games(["same"], ["same"])
    except ValueError as error:
        assert "held-out" in str(error)
    else:
        raise AssertionError("selection-complement gradients must not be called G_*")
