from omniopd.sampling import SamplingProtocol, stable_shuffled


def test_thinking_protocol_omits_ignored_temperature():
    SamplingProtocol("enabled", None, "high").validate(samples_per_state=3)
    try:
        SamplingProtocol("enabled", 0.7, "high").validate(samples_per_state=3)
    except ValueError as error:
        assert "temperature=None" in str(error)
    else:
        raise AssertionError("thinking mode must not pretend temperature is effective")


def test_n_sample_nonthinking_protocol_must_be_nondegenerate():
    SamplingProtocol("disabled", 0.7, None).validate(samples_per_state=3)
    try:
        SamplingProtocol("disabled", 0.0, None).validate(samples_per_state=3)
    except ValueError as error:
        assert "degenerate depth" in str(error)
    else:
        raise AssertionError("temperature-zero repeated calls do not test Monte Carlo depth")


def test_stable_shuffle_is_independent_of_upstream_iteration_order():
    assert stable_shuffled(["c", "a", "b"], seed=9) == stable_shuffled(
        ["b", "c", "a"], seed=9
    )
