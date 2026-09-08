from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
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
    if hasattr(value, "executed_action") and hasattr(value, "valid"):
        action = str(getattr(value, "executed_action", "")).strip()
        return {
            "action": action or None,
            "valid": bool(getattr(value, "valid", False)) and bool(action),
            "raw": getattr(value, "raw", None),
        }
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


def state_level_disagreement(
    student_action: str, teacher_samples: Iterable[Any]
) -> int | None:
    """Paper D(s): any valid Teacher draw differs; undefined if none is valid."""

    values = [
        value
        for value in (disagreement(student_action, sample) for sample in teacher_samples)
        if value is not None
    ]
    return int(any(values)) if values else None


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


@dataclass(frozen=True)
class ClusterBootstrapEstimate:
    estimate: float
    lower: float
    upper: float
    games: int
    replicates: int


@dataclass(frozen=True)
class ObservedSampleRatio:
    """Descriptive ratio on the realized sample, with equal game weight."""

    estimate: float
    games: int
    observed_rows: int
    estimand: str = "realized_selected_sample_game_balanced_ratio"


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


def observed_sample_game_balanced_ratio(
    rows: Iterable[dict[str, Any]],
    *,
    numerator_key: str,
    denominator_key: str,
) -> ObservedSampleRatio:
    """Return a descriptive conditional rate on the realized selected sample.

    This function deliberately does not use inclusion probabilities.  It is
    therefore useful for deterministic top-k samples, but it must not be
    interpreted as a finite-population estimate.  Games receive equal weight;
    within each game the realized sampled rows receive equal weight.
    """

    rows = list(rows)
    if not rows:
        raise ValueError("no observed rows for the requested ratio")
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(numerator_key) is None or row.get(denominator_key) is None:
            raise ValueError("missing outcomes make the observed-sample ratio undefined")
        by_game[str(row["game_id"])].append(row)
    numerator = 0.0
    denominator = 0.0
    for group in by_game.values():
        size = len(group)
        numerator += sum(float(row[numerator_key]) for row in group) / size
        denominator += sum(float(row[denominator_key]) for row in group) / size
    if denominator <= 0.0:
        raise ValueError("conditioning event has zero observed probability")
    return ObservedSampleRatio(
        estimate=numerator / denominator,
        games=len(by_game),
        observed_rows=len(rows),
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


def design_weighted_game_balanced_ratio(
    rows: Iterable[dict[str, Any]],
    *,
    numerator_key: str,
    denominator_key: str,
    population_states_by_game: dict[str, int],
    probability_key: str = "inclusion_probability",
) -> float:
    """Ratio of two game-balanced Horvitz–Thompson finite-population means."""

    rows = list(rows)
    if not rows or not population_states_by_game:
        raise ValueError("sample and population game counts are required")
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        game = str(row.get("game_id"))
        if game not in population_states_by_game:
            raise ValueError(f"sampled game {game!r} is absent from target population")
        if row.get(numerator_key) is None or row.get(denominator_key) is None:
            raise ValueError("missing outcomes make this SAGE probability nonidentified")
        probability = row.get(probability_key)
        try:
            probability = float(probability)
        except (TypeError, ValueError) as error:
            raise ValueError("valid design inclusion probabilities are required") from error
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError(
                "valid design inclusion probabilities are required; deterministic top-k "
                "does not identify a population probability"
            )
        by_game[game].append(row)
    if set(by_game) != set(population_states_by_game):
        raise ValueError("every target game must have sampled states")
    numerator = 0.0
    denominator = 0.0
    for game, population_size in population_states_by_game.items():
        if int(population_size) <= 0:
            raise ValueError("population state counts must be positive")
        numerator += sum(
            float(row[numerator_key]) / float(row[probability_key])
            for row in by_game[game]
        ) / int(population_size)
        denominator += sum(
            float(row[denominator_key]) / float(row[probability_key])
            for row in by_game[game]
        ) / int(population_size)
    if denominator <= 0.0:
        raise ValueError("conditioning event has zero estimated probability")
    return numerator / denominator


def game_cluster_bootstrap_design_ratio(
    rows: Iterable[dict[str, Any]],
    *,
    numerator_key: str,
    denominator_key: str,
    population_states_by_game: dict[str, int],
    replicates: int = 10_000,
    rng_seed: int = 42,
) -> ClusterBootstrapEstimate:
    """Resample games and recompute the design-weighted SAGE ratio."""

    rows = list(rows)
    point = design_weighted_game_balanced_ratio(
        rows,
        numerator_key=numerator_key,
        denominator_key=denominator_key,
        population_states_by_game=population_states_by_game,
    )
    if replicates <= 0 or len(population_states_by_game) < 2:
        raise ValueError("positive replicates and at least two game clusters are required")
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_game[str(row["game_id"])].append(row)
    games = sorted(population_states_by_game)
    rng = random.Random(rng_seed)
    draws = []
    attempts = 0
    while len(draws) < replicates and attempts < replicates * 20:
        attempts += 1
        sampled_rows = []
        sampled_population = {}
        for draw_index in range(len(games)):
            game = rng.choice(games)
            alias = f"bootstrap_{draw_index}"
            sampled_population[alias] = population_states_by_game[game]
            sampled_rows.extend(dict(row, game_id=alias) for row in by_game[game])
        try:
            draws.append(
                design_weighted_game_balanced_ratio(
                    sampled_rows,
                    numerator_key=numerator_key,
                    denominator_key=denominator_key,
                    population_states_by_game=sampled_population,
                )
            )
        except ValueError as error:
            if "conditioning event" not in str(error):
                raise
    if len(draws) != replicates:
        raise ValueError("conditioning event is too sparse for a stable cluster bootstrap")
    draws.sort()
    return ClusterBootstrapEstimate(
        point,
        draws[int(0.025 * (replicates - 1))],
        draws[int(0.975 * (replicates - 1))],
        len(games),
        replicates,
    )


def paired_game_cluster_bootstrap_indicator_difference(
    reference_rows: Iterable[dict[str, Any]],
    comparison_rows: Iterable[dict[str, Any]],
    *,
    indicator_key: str,
    population_states_by_game: dict[str, int] | None = None,
    probability_key: str = "inclusion_probability",
    replicates: int = 10_000,
    rng_seed: int = 42,
) -> ClusterBootstrapEstimate:
    """Estimate comparison-reference on common games with paired resampling.

    Without ``population_states_by_game`` this is the realized selected-sample,
    game-balanced contrast.  With population counts it is the difference of
    two game-balanced Horvitz--Thompson means and therefore requires a known,
    non-zero inclusion probability for every membership in both groups.
    """

    reference_rows = list(reference_rows)
    comparison_rows = list(comparison_rows)
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    by_reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_comparison: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows, destination in [
        (reference_rows, by_reference),
        (comparison_rows, by_comparison),
    ]:
        for row in rows:
            destination[str(row["game_id"])].append(row)
    common_games = sorted(set(by_reference) & set(by_comparison))
    if len(common_games) < 2:
        raise ValueError(
            "paired group contrast requires at least two common game clusters"
        )

    def game_mean(rows: list[dict[str, Any]], game: str) -> float:
        if any(row.get(indicator_key) is None for row in rows):
            raise ValueError(
                "missing outcomes on common games make the paired group contrast "
                "nonidentified"
            )
        if population_states_by_game is None:
            return sum(float(row[indicator_key]) for row in rows) / len(rows)
        if game not in population_states_by_game:
            raise ValueError(
                f"common game {game!r} is absent from the target population"
            )
        population_size = int(population_states_by_game[game])
        if population_size <= 0:
            raise ValueError("population state counts must be positive")
        weighted_total = 0.0
        for row in rows:
            try:
                probability = float(row[probability_key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "valid design inclusion probabilities are required for the "
                    "population group contrast"
                ) from error
            if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
                raise ValueError(
                    "valid design inclusion probabilities are required; deterministic "
                    "top-k does not identify a population group contrast"
                )
            weighted_total += float(row[indicator_key]) / probability
        return weighted_total / population_size

    game_differences = [
        game_mean(by_comparison[game], game)
        - game_mean(by_reference[game], game)
        for game in common_games
    ]
    point = sum(game_differences) / len(game_differences)
    rng = random.Random(rng_seed)
    draws = sorted(
        sum(rng.choice(game_differences) for _ in common_games)
        / len(common_games)
        for _ in range(replicates)
    )
    return ClusterBootstrapEstimate(
        point,
        draws[int(0.025 * (replicates - 1))],
        draws[int(0.975 * (replicates - 1))],
        len(common_games),
        replicates,
    )
