from __future__ import annotations

from typing import Any

from .representativeness import fixed_width_bin
from .schema import RolloutTurn


def state_diagnostic_features(
    turn: RolloutTurn,
    *,
    previous_observation: str | None,
    max_steps: int,
    depth_bins: int = 10,
    admissible_bins: int = 10,
    max_admissible: int = 50,
) -> dict[str, Any]:
    """Return the preregistered marginal M2 diagnostics for one state."""

    if max_steps <= 0 or depth_bins <= 0 or admissible_bins <= 0 or max_admissible <= 0:
        raise ValueError("M2 feature binning parameters must be positive")
    words = turn.student.executed_action.split()
    return {
        "task_type": turn.state.task_type,
        "turn_depth_bin": fixed_width_bin(
            float(turn.state.turn_index),
            minimum=0.0,
            maximum=float(max_steps),
            bins=depth_bins,
        ),
        "action_type": words[0].lower() if words else "none",
        "n_admissible_bin": fixed_width_bin(
            float(len(turn.state.admissible_actions)),
            minimum=0.0,
            maximum=float(max_admissible),
            bins=admissible_bins,
        ),
        "technical": int(not turn.student.valid),
        "empty_response": int(not turn.student.raw.strip()),
        "repeat_observation": int(
            previous_observation is not None
            and turn.state.observation == previous_observation
        ),
    }
