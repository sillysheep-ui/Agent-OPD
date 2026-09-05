def test_named_regression_recovers_group_interaction_without_group_mixing():
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        from unittest import SkipTest

        raise SkipTest("optional numpy dependency is not installed")

    from omniopd.analysis.m3_stats import fit_transfer_regression

    rows = []
    for group in ["A1", "A3"]:
        for index in range(30):
            source = (index - 14.5) / 10
            indicator = float(group == "A3")
            sim = ((index * 7) % 13) / 13
            technical = index % 2
            repeat = (index // 2) % 2
            outcome = (
                2.0
                + 3.0 * indicator
                + 4.0 * source
                + 5.0 * source * indicator
                + 0.7 * sim
                - 0.2 * technical
                + 0.1 * repeat
            )
            rows.append(
                {
                    "group": group,
                    "gamefile": f"g{index % 10}",
                    "S_i": source,
                    "T_i": outcome,
                    "sim_topK": sim,
                    "technical": technical,
                    "repeat_obs": repeat,
                    "task_type": "task",
                    "action_type": "move",
                }
            )
    result = fit_transfer_regression(rows)
    assert abs(result.coefficients["group_A3"] - 3.0) < 1e-9
    assert abs(result.coefficients["S_x_group_A3"] - 5.0) < 1e-9
