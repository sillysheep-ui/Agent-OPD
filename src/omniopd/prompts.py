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

# v2 fixes exactly the two problems the first pilot exposed:
#   1. nothing told the model to track task progress, so it lost the subgoal it
#      had already reached and re-walked the same receptacles;
#   2. nothing told the model what to do when an action changed nothing, so it
#      repeated one command (observed up to 44 identical `look` turns).
# Command syntax was deliberately left alone: every invalid action in the
# pilot was not_admissible, never a malformed command.
STUDENT_SYSTEM_PROMPT_V2 = """You are an ALFWorld household agent.

Complete the given household task by interacting with the environment one step at a time.

At each turn, use the task description, interaction history, current observation, and current admissible actions to choose the next action.

Track the task state as you go: which target object or objects you still need, where you last saw them, what you are carrying, and which subgoals such as finding, taking, cleaning, heating, cooling, opening or placing are already done. Finish any required clean, heat or cool step before the final placement, and for a two-object task handle two distinct objects.

If your recent actions did not change the observation, do not repeat them. Choose a different admissible action that gains information or makes progress.

Return exactly one action from the current admissible actions in this format:
Action: <command>

Use the action exactly as written in the admissible action list. Do not change object names, object numbers, or command syntax.

Do not output reasoning, explanations, multiple actions, predicted observations, or future turns."""

# "react" is the prompt used by the reference ALFWorld SFT protocol
# (qiangwhu/Qwen3-ALFWorld-FullHistory-SFT, as carried by
# legacy/main_document/collect_teacher_rollout.py), kept verbatim so a
# reproduction attempt can be told apart from our own minimal prompt.
STUDENT_SYSTEM_PROMPT_REACT = """You are an ALFWorld household agent. Solve the current task one environment turn at a time.
Reply in this format:
Thought: <brief reasoning>
Action: <one executable ALFWorld command>
Use the command grammar shown by the environment (for example go to, take, move, open, close, use, clean, heat, cool).
To place objects use "move X to Y" (NOT "put X in/on Y"). Object and receptacle names include their numbers, e.g. "fridge 1", "drawer 2".
Do not simulate future observations or future turns."""

STUDENT_SYSTEM_PROMPTS: dict[str, str] = {
    "v1": STUDENT_SYSTEM_PROMPT,
    "v2": STUDENT_SYSTEM_PROMPT_V2,
    "react": STUDENT_SYSTEM_PROMPT_REACT,
}
STUDENT_SYSTEM_PROMPT_DEFAULT = "v1"


def resolve_student_prompt(name: str | None = None) -> tuple[str, str]:
    """Return (prompt_name, prompt_text) for an explicit prompt version.

    The prompt is part of the protocol, so every artifact records which
    version produced it instead of silently reading a module constant.
    """

    key = STUDENT_SYSTEM_PROMPT_DEFAULT if name is None else str(name)
    if key not in STUDENT_SYSTEM_PROMPTS:
        raise ValueError(
            f"unknown Student prompt version {key!r}; "
            f"known versions are {sorted(STUDENT_SYSTEM_PROMPTS)}"
        )
    return key, STUDENT_SYSTEM_PROMPTS[key]


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
