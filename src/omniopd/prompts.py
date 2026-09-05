from __future__ import annotations

from collections.abc import Sequence

from .schema import Message

STUDENT_SYSTEM_PROMPT = """You are an ALFWorld household agent.

Complete the given household task by interacting with the environment one step at a time.

At each turn, use the task description, interaction history, current observation, and current admissible actions to choose the next action.

Return exactly one action from the current admissible actions in this format:
Action: <command>

Use the action exactly as written in the admissible action list. Do not change object names, object numbers, or command syntax.

Do not output reasoning, explanations, multiple actions, predicted observations, or future turns."""

TEACHER_SYSTEM_PROMPT = """You are an expert ALFWorld household agent acting as a correction teacher.

Use the supplied task, executed interaction history, current observation, and current admissible actions. Choose exactly one admissible next action.

Return exactly:
Action: <command>

Do not output reasoning, explanations, multiple actions, predicted observations, or future turns."""


def admissible_block(actions: Sequence[str]) -> str:
    return "\n".join(f"- {action}" for action in actions)


def initial_user_message(task: str, observation: str, actions: Sequence[str]) -> str:
    return (
        f"Task:\n{task.strip()}\n\n"
        f"Observation:\n{observation.strip()}\n\n"
        f"Admissible actions:\n{admissible_block(actions)}"
    )


def turn_user_message(observation: str, actions: Sequence[str]) -> str:
    return (
        f"Observation:\n{observation.strip()}\n\n"
        f"Admissible actions:\n{admissible_block(actions)}"
    )


def replace_system(messages: Sequence[Message], system_prompt: str) -> list[Message]:
    """Reuse exactly z_t while changing only the policy prompt P."""

    if not messages or messages[0].get("role") != "system":
        raise ValueError("messages must start with a system message")
    copied = [dict(message) for message in messages]
    copied[0] = {"role": "system", "content": system_prompt}
    return copied
