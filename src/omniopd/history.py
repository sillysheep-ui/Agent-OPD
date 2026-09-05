from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .prompts import initial_user_message, turn_user_message
from .schema import AgentState, Message


@dataclass
class ConversationHistory:
    """A fail-closed history state machine with no stale-observation path."""

    task: str
    game_id: str
    task_type: str
    _messages: list[Message]
    _observation: str
    _admissible_actions: tuple[str, ...]
    _turn_index: int = 0
    _awaiting_observation: bool = False

    @classmethod
    def start(
        cls,
        *,
        system_prompt: str,
        task: str,
        initial_observation: str,
        admissible_actions: Sequence[str],
        game_id: str,
        task_type: str,
    ) -> "ConversationHistory":
        actions = tuple(str(action) for action in admissible_actions)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": initial_user_message(task, initial_observation, actions),
            },
        ]
        return cls(task, game_id, task_type, messages, initial_observation, actions)

    @property
    def messages(self) -> list[Message]:
        return [dict(message) for message in self._messages]

    def snapshot(
        self,
        messages: Sequence[Message] | None = None,
        *,
        truncation: dict | None = None,
        state_source: str = "student",
    ) -> AgentState:
        if self._awaiting_observation:
            raise RuntimeError("cannot snapshot between an action and its resulting observation")
        current = self.messages if messages is None else [dict(message) for message in messages]
        if not current or current[-1].get("role") != "user":
            raise ValueError("a model-query state must end with the current user observation")
        return AgentState(
            task=self.task,
            observation=self._observation,
            admissible_actions=self._admissible_actions,
            messages=tuple(current),
            game_id=self.game_id,
            task_type=self.task_type,
            turn_index=self._turn_index,
            full_messages=tuple(self.messages),
            truncation=dict(truncation or {}),
            state_source=state_source,
        )

    def record_action(self, executed_action: str) -> None:
        if self._awaiting_observation:
            raise RuntimeError("previous action has no recorded environment result")
        self._messages.append({"role": "assistant", "content": f"Action: {executed_action}"})
        self._awaiting_observation = True

    def record_observation(
        self,
        observation: str,
        admissible_actions: Sequence[str],
        *,
        done: bool,
    ) -> None:
        if not self._awaiting_observation:
            raise RuntimeError("an observation must follow an executed action")
        self._turn_index += 1
        self._observation = str(observation)
        self._admissible_actions = tuple(str(action) for action in admissible_actions)
        self._awaiting_observation = False
        if not done:
            self._messages.append(
                {
                    "role": "user",
                    "content": turn_user_message(self._observation, self._admissible_actions),
                }
            )

    def assert_query_ready(self) -> None:
        if self._awaiting_observation or self._messages[-1].get("role") != "user":
            raise RuntimeError("history is not ready for a model query")
        expected = ["system", "user"]
        for index in range(2, len(self._messages)):
            expected.append("assistant" if index % 2 == 0 else "user")
        roles = [message.get("role") for message in self._messages]
        if roles != expected:
            raise RuntimeError(f"malformed role sequence: {roles}")
