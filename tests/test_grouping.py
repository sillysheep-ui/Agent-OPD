from omniopd.dataset import build_training_rows
from omniopd.grouping import group_training_rows, state_group_batches
from omniopd.schema import ActionSample, AgentState, CorrectionRecord


def test_n_actions_remain_one_state_sampling_unit():
    state = AgentState(
        "t",
        "o",
        ("a", "b", "c"),
        ({"role": "system", "content": "s"}, {"role": "user", "content": "u"}),
        "g",
        "x",
        2,
    )
    student = ActionSample("Action: a", "a", True, True, None)
    teachers = [ActionSample(f"Action: {a}", a, True, True, None) for a in ["a", "b", "c"]]
    rows = build_training_rows(
        [CorrectionRecord(state, student, teachers, "random", 1.0, 3)],
        weighting="state_mean",
    )
    groups = group_training_rows(rows)
    assert len(groups) == 1
    assert len(groups[0].rows) == 3
    assert abs(groups[0].state_weight - 1.0) < 1e-12
    assert len(state_group_batches(groups, states_per_batch=1)) == 1
