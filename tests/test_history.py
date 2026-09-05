from omniopd.history import ConversationHistory
from omniopd.prompts import STUDENT_SYSTEM_PROMPT
from omniopd.context import TaskPreservingTruncator


def test_history_uses_real_initial_observation_and_new_transition_observation():
    history = ConversationHistory.start(
        system_prompt=STUDENT_SYSTEM_PROMPT,
        task="put apple away",
        initial_observation="You are in the kitchen.",
        admissible_actions=["look", "go to counter 1"],
        game_id="g1",
        task_type="pick_and_place",
    )
    assert "Task:\nput apple away" in history.messages[1]["content"]
    assert "Observation:\nYou are in the kitchen." in history.messages[1]["content"]
    history.record_action("look")
    history.record_observation("A red apple is visible.", ["take apple 1"], done=False)
    history.assert_query_ready()
    assert [m["role"] for m in history.messages] == ["system", "user", "assistant", "user"]
    assert "A red apple is visible." in history.messages[-1]["content"]
    assert history.messages[-1]["content"].count("You are in the kitchen.") == 0


def test_snapshot_cannot_be_taken_between_action_and_observation():
    history = ConversationHistory.start(
        system_prompt="s",
        task="t",
        initial_observation="o0",
        admissible_actions=["look"],
        game_id="g",
        task_type="x",
    )
    history.record_action("look")
    try:
        history.snapshot()
    except RuntimeError:
        pass
    else:
        raise AssertionError("snapshot should fail while the environment transition is pending")


class LengthTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(message["content"]) for message in messages)))


class ThinkingAwareTokenizer:
    def __init__(self):
        self.observed = []

    def apply_chat_template(self, messages, **kwargs):
        self.observed.append(kwargs["enable_thinking"])
        return list(range(len(messages)))


def test_truncation_uses_the_behavior_policys_thinking_template():
    tokenizer = ThinkingAwareTokenizer()
    truncator = TaskPreservingTruncator(tokenizer, enable_thinking=True)
    truncator.truncate(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]
    )
    assert tokenizer.observed and all(tokenizer.observed)


def test_truncation_never_falls_back_to_the_initial_observation_when_current_state_is_too_large():
    history = ConversationHistory.start(
        system_prompt="s",
        task="t",
        initial_observation="o0",
        admissible_actions=["look"],
        game_id="g",
        task_type="x",
    )
    history.record_action("look")
    history.record_observation("x" * 200, ["look"], done=False)
    truncator = TaskPreservingTruncator(LengthTokenizer(), max_context_tokens=100, reserve_tokens=10)
    try:
        truncator.truncate(history.messages)
    except ValueError as error:
        assert "current action/observation" in str(error)
    else:
        raise AssertionError("an oversized current state must fail rather than becoming stale")
