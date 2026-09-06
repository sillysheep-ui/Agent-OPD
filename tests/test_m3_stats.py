def test_named_regression_recovers_checkpoint_effect_in_crossed_panels():
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        from unittest import SkipTest

        raise SkipTest("optional numpy dependency is not installed")

    from omniopd.analysis.m3_stats import fit_transfer_regression

    rows = []
    for panel in ["A1", "A3"]:
        for index in range(30):
            for checkpoint in ["A1", "A3"]:
                c = float(checkpoint == "A3")
                p = float(panel == "A3")
                source = (index - 14.5) / 10 + 0.13 * c + 0.07 * p
                sim = ((index * 7) % 13) / 13
                technical = index % 2
                repeat = (index // 2) % 2
                outcome = (
                    2.0 + 3.0 * c + 1.5 * p + 0.4 * c * p + 4.0 * source
                    + 5.0 * source * c + 0.8 * source * p
                    + 0.6 * source * c * p + 0.7 * sim
                    - 0.2 * technical + 0.1 * repeat
                )
                rows.append(
                    {
                        "source_id": f"{panel}-{index}",
                        "checkpoint_group": checkpoint,
                        "panel_group": panel,
                        "checkpoint_experiment": {"A1": "random", "A3": "entropy"}[checkpoint],
                        "panel_experiment": {"A1": "random", "A3": "entropy"}[panel],
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
    assert abs(result.coefficients["checkpoint_A3"] - 3.0) < 1e-9
    assert abs(result.coefficients["S_x_checkpoint_A3"] - 5.0) < 1e-9
