import json

from omniopd.opd_adapter import (
    align_teacher_sampled_token_logprobs,
    audit_shared_token_id_space,
    build_fixed_pool_opd_prompts,
    remap_teacher_scores_to_student_layout,
)
from omniopd.prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT
from omniopd.schema import ActionSample, AgentState, RolloutTurn


def _turn(game_id="game-1", *, state_source="student"):
    state = AgentState(
        task="put book on desk",
        observation="A book is here.",
        admissible_actions=("take book", "look"),
        messages=(
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": "Task: put book on desk.\nAction choices: take book; look"},
        ),
        game_id=game_id,
        task_type="pick_and_place",
        turn_index=0,
        state_source=state_source,
    )
    sample = ActionSample(
        raw="Action: look", executed_action="look", valid=True,
        had_action_marker=True, failure_reason=None,
    )
    return RolloutTurn(state=state, student=sample)


def _selection(turn):
    return {
        "protocol_version": "omniopd-v1",
        "state_hash": turn.state.state_hash,
        "game_id": turn.state.game_id,
        "turn_index": turn.state.turn_index,
        "selection_policy": "uniform_per_game_nested_v1",
    }


def _raises_value_error(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_fixed_pool_opd_prompt_preserves_two_policy_views_without_behavior_action():
    first, second = _turn("game-1"), _turn("game-2")
    rows = build_fixed_pool_opd_prompts(
        [first, second], [_selection(first), _selection(second)]
    )
    assert len(rows) == 2
    assert rows[0]["prompt"][0]["content"] == STUDENT_SYSTEM_PROMPT
    assert rows[0]["extra_info"]["teacher_prompt"][0]["content"] == TEACHER_SYSTEM_PROMPT
    assert rows[0]["extra_info"]["admissible_actions"] == ["take book", "look"]
    assert rows[0]["prompt"][1:] == rows[0]["extra_info"]["teacher_prompt"][1:]
    assert rows[0]["extra_info"]["state_weight"] == 1.0
    assert "Action: look" not in json.dumps(rows[0], ensure_ascii=False)


def test_fixed_pool_opd_prompt_rejects_duplicate_and_wrong_state_identity():
    turn = _turn()
    selection = _selection(turn)
    _raises_value_error(build_fixed_pool_opd_prompts, [turn], [selection, selection])
    wrong_game = dict(selection, game_id="other-game")
    _raises_value_error(build_fixed_pool_opd_prompts, [turn], [wrong_game])
    teacher_state = _turn(state_source="teacher")
    _raises_value_error(
        build_fixed_pool_opd_prompts, [teacher_state], [_selection(teacher_state)]
    )


class _Backend:
    def __init__(self, rules):
        self.rules = rules

    def to_str(self):
        return json.dumps(self.rules)


class _Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = None
    all_special_ids = [0, 1, 2]
    chat_template = "{{ messages }}"

    def __init__(self, vocab=None, rules=None, chat_template=None):
        self.vocab = vocab or {"<pad>": 0, "<bos>": 1, "<eos>": 2, "Action": 3}
        self.backend_tokenizer = _Backend(rules or {"model": "bpe", "decoder": "byte"})
        self.chat_template = chat_template or self.chat_template

    def get_vocab(self):
        return self.vocab


def test_opd_tokenizer_audit_requires_identical_id_meanings_and_rules():
    student, teacher = _Tokenizer(), _Tokenizer()
    audit = audit_shared_token_id_space(student, teacher)
    assert audit["vocab_size"] == 4
    _raises_value_error(
        audit_shared_token_id_space,
        student,
        _Tokenizer(vocab={"<pad>": 0, "<bos>": 1, "<eos>": 2, "Action": 4}),
    )
    _raises_value_error(
        audit_shared_token_id_space,
        student,
        _Tokenizer(rules={"model": "wordpiece", "decoder": "byte"}),
    )
    different_template = audit_shared_token_id_space(
        student, _Tokenizer(chat_template="{{ different }}")
    )
    assert different_template["chat_templates_equal"] is False
    assert (
        different_template["student_chat_template_sha256"]
        != different_template["teacher_chat_template_sha256"]
    )


def test_teacher_scores_slice_only_student_response_and_reject_bad_alignment():
    kwargs = {
        "teacher_prompt_ids": [11, 12, 13],
        "student_response_ids": [21, 22],
        "scored_sequence_ids": [11, 12, 13, 21, 22],
        "scored_token_logprobs": [None, -1.0, -2.0, -3.0, -4.0],
    }
    assert align_teacher_sampled_token_logprobs(**kwargs) == [-3.0, -4.0]
    _raises_value_error(
        align_teacher_sampled_token_logprobs,
        **dict(kwargs, scored_sequence_ids=[11, 12, 13, 22, 21]),
    )
    _raises_value_error(
        align_teacher_sampled_token_logprobs,
        **dict(kwargs, scored_token_logprobs=[None, -1.0, -2.0, None, -4.0]),
    )


def test_teacher_scores_remap_distinct_prompt_lengths_for_verl_estimator():
    kwargs = {
        "student_prompt_ids": [1, 2],
        "teacher_prompt_ids": [3, 4, 5],
        "student_response_ids": [6, 7],
        # veRL stores the score/ID of the next token at the current position.
        "teacher_scored_ids": [[4], [5], [6], [7], [0]],
        "teacher_scored_logprobs": [[-0.1], [-0.2], [-1.5], [-2.5], [0.0]],
        "pad_token_id": 0,
    }
    ids, scores = remap_teacher_scores_to_student_layout(**kwargs)
    assert ids == [[0], [6], [7], [0]]
    assert scores == [[0.0], [-1.5], [-2.5], [0.0]]
    # veRL slices one position before the response: prompt_len - 1.
    assert scores[len(kwargs["student_prompt_ids"]) - 1 : -1] == [[-1.5], [-2.5]]
    _raises_value_error(
        remap_teacher_scores_to_student_layout,
        **dict(kwargs, teacher_scored_ids=[[4], [5], [7], [6], [0]]),
    )
    _raises_value_error(
        remap_teacher_scores_to_student_layout,
        **dict(kwargs, teacher_scored_logprobs=[[0], [0], [None], [-2.5], [0]]),
    )
    _raises_value_error(
        remap_teacher_scores_to_student_layout,
        **dict(kwargs, teacher_scored_logprobs=[[0], [0], [-1.5, -1.0], [-2.5], [0]]),
    )
