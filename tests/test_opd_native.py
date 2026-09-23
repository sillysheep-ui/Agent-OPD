"""Contracts for the veRL-native OPD overrides (see src/omniopd/opd_native.py).

Two things must be exactly right, and both are pure functions here:

* the Teacher prompt is rendered with the Teacher's own template, which makes it
  longer than the Student's (the empty thinking block), and
* the Teacher's scores land where veRL reads them, i.e. the score of response
  token ``i`` at sequence position ``student_prompt_length - 1 + i``.
"""

from omniopd.opd_native_layout import (
    render_prompt,
    teacher_messages,
    teacher_rows_for_student_layout,
)


class _FakeTokenizer:
    def __init__(self, extra=()):
        self.extra = list(extra)
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        self.calls.append((tokenize, add_generation_prompt, enable_thinking))
        return [[*messages[0]["ids"], *self.extra]]


def test_teacher_messages_needs_a_state_history():
    history = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert teacher_messages({"raw_prompt": history}) == history
    for bad in (
        None,
        {},
        {"raw_prompt": []},
        {"raw_prompt": [{"role": "user", "content": "u"}]},
        {"raw_prompt": [{"role": "system", "content": "s"}]},
    ):
        try:
            teacher_messages(bad)
        except ValueError:
            continue
        raise AssertionError(f"malformed sample was accepted: {bad}")


def test_render_prompt_uses_the_teachers_own_template_with_thinking_off():
    tokenizer = _FakeTokenizer(extra=[7, 8])
    rendered = render_prompt(tokenizer, [{"role": "system", "content": "s", "ids": [1, 2]}])
    assert rendered == [1, 2, 7, 8]
    assert tokenizer.calls[-1] == (True, True, False)


def test_teacher_scores_land_on_the_student_layout():
    student_prompt = [11, 12, 13, 14, 15]           # length 5
    teacher_prompt = [11, 12, 13, 14, 15, 90, 91]   # length 7: thinking block
    response = [21, 22, 23]
    scored = teacher_prompt + response
    # veRL stores the id of token i+1 at row i and appends a dummy final row.
    teacher_ids = [[token] for token in scored[1:] + [0]]
    teacher_logprobs = [[float(index)] for index in range(len(scored))]

    rows, scores = teacher_rows_for_student_layout(
        student_prompt=student_prompt,
        teacher_prompt=teacher_prompt,
        response=response,
        teacher_ids=teacher_ids,
        teacher_logprobs=teacher_logprobs,
        pad_token_id=0,
    )

    assert len(rows) == len(student_prompt) + len(response)
    assert [row[0] for row in rows] == [0, 0, 0, 0, 21, 22, 23, 0]
    # The Teacher's rows 6, 7, 8 hold the scores of response tokens 0, 1, 2.
    assert [row[0] for row in scores] == [0.0, 0.0, 0.0, 0.0, 6.0, 7.0, 8.0, 0.0]


def test_layout_rejects_a_teacher_that_does_not_cover_the_response():
    for teacher_ids, teacher_logprobs in (
        ([[1], [2]], [[0.0], [0.0]]),                # too short
        ([[1], [2], [3], [0]], [[0.0], [1.0, 2.0], [0.0], [0.0]]),  # K columns
    ):
        try:
            teacher_rows_for_student_layout(
                student_prompt=[1, 2],
                teacher_prompt=[1, 2],
                response=[3],
                teacher_ids=teacher_ids,
                teacher_logprobs=teacher_logprobs,
                pad_token_id=0,
            )
        except ValueError:
            continue
        raise AssertionError(f"malformed Teacher rows were accepted: {teacher_ids}")
