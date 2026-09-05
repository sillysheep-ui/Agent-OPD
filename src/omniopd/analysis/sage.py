from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from ..provenance import sha256_json
from ..schema import AgentState

SAGE_LABELS = frozenset({"Skip", "Weak", "Strong"})
SAGE_JUDGE_SYSTEM_PROMPT = """You are an expert ALFWorld intervention judge.

Given only the task, executed history, current observation, current admissible actions, and the Student's chosen action, assess whether a corrective intervention is needed.

Skip: the Student action is reasonable and needs no correction.
Weak: the action is questionable or suboptimal, so correction may help.
Strong: the action is clearly wrong or harmful, so correction is necessary.

Return exactly one word: Skip, Weak, or Strong."""


def build_sage_judge_messages(
    state: AgentState, student_action: str
) -> list[dict[str, str]]:
    """Build a blind judge query containing no Teacher output or selection proxy."""

    visible_state = {
        "task": state.task,
        "executed_history_and_current_state": [
            dict(message) for message in state.messages[1:]
        ],
        "current_observation": state.observation,
        "current_admissible_actions": list(state.admissible_actions),
        "student_action": str(student_action),
    }
    import json

    return [
        {"role": "system", "content": SAGE_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                visible_state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        },
    ]


def sage_judge_query_sha256(state: AgentState, student_action: str) -> str:
    return sha256_json(build_sage_judge_messages(state, student_action))


def parse_sage_label(response: str) -> str | None:
    value = str(response or "").strip()
    return value if value in SAGE_LABELS else None


def normalize_teacher_sample(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        action = value.get("executed_action") or value.get("teacher_action") or value.get("action")
        normalized_action = str(action).strip() if action is not None else None
        valid = bool(
            value.get("valid", value.get("teacher_valid", normalized_action is not None))
        ) and bool(normalized_action)
        return {
            "action": normalized_action,
            "valid": valid,
            "raw": value.get("raw") or value.get("teacher_raw"),
        }
    if value is None:
        return {"action": None, "valid": False, "raw": None}
    action = str(value).strip()
    return {"action": action or None, "valid": bool(action), "raw": None}


def disagreement(student_action: str, teacher_sample: Any) -> int | None:
    sample = normalize_teacher_sample(teacher_sample)
    if not sample["valid"] or sample["action"] is None:
        return None
    return int(str(student_action) != sample["action"])


def deterministic_or_judged_label(
    *, student_valid: bool, judge_label: str | None
) -> str | None:
    """Resolve a SAGE label without imputing failed blind-judge calls."""

    if not student_valid:
        return "Strong"
    if judge_label is None:
        return None
    if judge_label not in SAGE_LABELS:
        raise ValueError("judge_label must be an exact SAGE label or missing")
    return judge_label


@dataclass(frozen=True)
class GameBalancedProportion:
    estimate: float
    games: int
    observed_states: int
    missing_states: int


@dataclass(frozen=True)
class DesignWeightedProportion:
    estimate: float
    games: int
    sampled_states: int
    estimand: str = "finite_population_game_balanced_mean"


def game_balanced_indicator(
    rows: Iterable[dict[str, Any]], *, indicator_key: str
) -> GameBalancedProportion:
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing = 0
    for row in rows:
        value = row.get(indicator_key)
        if value is None:
            missing += 1
            continue
        by_game[str(row["game_id"])].append(row)
    if not by_game:
        raise ValueError("no observed states for the requested indicator")
    game_means = [
        sum(float(row[indicator_key]) for row in group) / len(group) for group in by_game.values()
    ]
    return GameBalancedProportion(
        sum(game_means) / len(game_means),
        len(game_means),
        sum(len(group) for group in by_game.values()),
        missing,
    )


def design_weighted_game_balanced_indicator(
    rows: Iterable[dict[str, Any]],
    *,
    indicator_key: str,
    population_states_by_game: dict[str, int],
    probability_key: str = "inclusion_probability",
) -> DesignWeightedProportion:
    """Horvitz-Thompson estimate for a sampled finite state population.

    This is the appropriate overall SAGE summary when states were sampled with
    known, non-zero inclusion probabilities. Deterministic top-k samples have
    no design-based population estimator and are rejected.
    """

    rows = list(rows)
    if not population_states_by_game:
        raise ValueError("population state counts are required")
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        game = str(row["game_id"])
        if game not in population_states_by_game:
            raise ValueError(f"sampled game {game!r} is absent from the target population")
        if row.get(indicator_key) is None:
            raise ValueError("all sampled states need observed SAGE labels")
        raw_probability = row.get(probability_key)
        probability = float(raw_probability) if raw_probability is not None else float("nan")
        if not 0.0 < probability <= 1.0:
            raise ValueError(
                "valid inclusion probabilities are required; deterministic top-k cannot "
                "identify a population proportion"
            )
        by_game[game].append(row)
    if set(by_game) != set(population_states_by_game):
        missing = sorted(set(population_states_by_game) - set(by_game))
        raise ValueError(f"target games without sampled states: {missing[:5]}")

    game_estimates = []
    for game in sorted(population_states_by_game):
        population_size = int(population_states_by_game[game])
        if population_size <= 0:
            raise ValueError(f"game {game!r} has a non-positive population size")
        estimated_total = sum(
            float(row[indicator_key]) / float(row[probability_key]) for row in by_game[game]
        )
        game_estimates.append(estimated_total / population_size)
    return DesignWeightedProportion(
        estimate=sum(game_estimates) / len(game_estimates),
        games=len(game_estimates),
        sampled_states=len(rows),
    )
