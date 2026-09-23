"""Contracts for the one hook we add to veRL's native OPD path.

The hook exists because the Teacher prompt must be rendered with the Teacher's
own tokenizer and template, which makes it *longer* than the Student's prompt
(the empty thinking block). The score remap is the part that must be exactly
right: veRL reads the Teacher score for response token i at sequence position
``student_prompt_length - 1 + i``, so the Teacher's own slice has to start at
``teacher_prompt_length - 1``. The helpers are pure, so this contract is checked
without torch, a GPU, a Ray worker or a running Teacher.
"""

from omniopd.opd_adapter import (
    remap_teacher_scores_to_student_layout,
    render_teacher_prompt_ids,
    teacher_state_messages,
)


class _FakeTokenizer:
    """Chat-template stand-in that appends a thinking block, like the Teacher."""

    def __init__(self, extra=()):
        self.extra = list(extra)
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        self.calls.append((tokenize, add_generation_prompt, enable_thinking))
        return [[*messages[0]["ids"], *self.extra]]


def test_teacher_state_messages_requires_a_state_history():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert teacher_state_messages({"raw_prompt": messages}) == messages
    for bad in (
        None,
        {},
        {"raw_prompt": []},
        {"raw_prompt": [{"role": "user", "content": "u"}]},
        {"raw_prompt": [{"role": "system", "content": "s"}]},
    ):
        try:
            teacher_state_messages(bad)
        except ValueError:
            continue
        raise AssertionError(f"malformed sample was accepted: {bad}")


def test_teacher_prompt_is_rendered_by_the_teacher_template():
    tokenizer = _FakeTokenizer(extra=[7, 8])
    messages = [{"role": "system", "content": "s", "ids": [1, 2]}]
    rendered = render_teacher_prompt_ids(tokenizer, messages)
    assert rendered == [1, 2, 7, 8]
    assert tokenizer.calls[-1] == (True, True, False)  # tokenize, add_generation, thinking off


def test_teacher_rows_land_on_the_student_layout_when_prompts_differ():
    student_prompt = [11, 12, 13, 14, 15]           # length 5
    teacher_prompt = [11, 12, 13, 14, 15, 90, 91]   # length 7: the thinking block
    response = [21, 22, 23]
    scored = teacher_prompt + response
    # veRL stores the id of token i+1 at row i and appends a dummy final row.
    scored_ids = [[token] for token in scored[1:] + [0]]
    scored_logprobs = [[float(index)] for index in range(len(scored))]

    rows, scores = remap_teacher_scores_to_student_layout(
        student_prompt_ids=student_prompt,
        teacher_prompt_ids=teacher_prompt,
        student_response_ids=response,
        teacher_scored_ids=scored_ids,
        teacher_scored_logprobs=scored_logprobs,
        pad_token_id=0,
    )

    assert len(rows) == len(student_prompt) + len(response)
    assert [row[0] for row in rows] == [0, 0, 0, 0, 21, 22, 23, 0]
    # The Teacher's rows 6, 7, 8 hold the scores of response tokens 0, 1, 2.
    assert [row[0] for row in scores] == [0.0, 0.0, 0.0, 0.0, 6.0, 7.0, 8.0, 0.0]


def test_remap_rejects_a_topk_teacher_output():
    try:
        remap_teacher_scores_to_student_layout(
            student_prompt_ids=[1, 2],
            teacher_prompt_ids=[1, 2],
            student_response_ids=[3],
            teacher_scored_ids=[[3], [0]],
            teacher_scored_logprobs=[[1.0, 2.0], [0.0, 0.0]],
            pad_token_id=0,
        )
    except ValueError:
        return
    raise AssertionError("a K-column Teacher output was accepted by the width-1 remap")
