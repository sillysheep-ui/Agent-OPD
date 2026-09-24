from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

from .context import TaskPreservingTruncator
from .history import ConversationHistory
from .parser import parse_action
from .prompts import STUDENT_SYSTEM_PROMPT, replace_system
from .protocol import AgentEnvironment, ChatPolicy, GenerationSettings, ResetResult
from .schema import AgentState


@dataclass(frozen=True)
class BranchOutcome:
    intervention_action: str
    won: bool
    done_after_intervention: bool
    continuation_steps: int
    executed_actions: tuple[str, ...]


@dataclass(frozen=True)
class CounterfactualPair:
    student_branch: BranchOutcome
    teacher_branch: BranchOutcome

    @property
    def consequence(self) -> int:
        return int(self.teacher_branch.won) - int(self.student_branch.won)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summarize_position_counterfactuals(
    rows: Iterable[Mapping[str, Any]],
    *,
    population_states_by_game: Mapping[str, int],
    expected_state_hashes: Iterable[str],
    bootstrap_replicates: int = 50_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Summarize paired Position outcomes without treating draws as states.

    Multiple valid Teacher draws at one state are averaged first.  Known state
    inclusion probabilities then identify Horvitz--Thompson numerator and
    denominator totals within each game; games receive equal weight.  The
    primary paper estimand conditions on a valid Teacher draw that differs
    from the Student action.  The unconditional effect is retained only as a
    secondary diagnostic.
    The bootstrap resamples whole games and always keeps Student/Teacher
    outcomes paired.  If a selected state has no replayable Teacher action, or
    its design probability is unavailable, no population estimate is emitted.
    """

    if bootstrap_replicates <= 0 or bootstrap_seed < 0:
        raise ValueError("bootstrap replicates must be positive and seed non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    population = {str(game): int(count) for game, count in population_states_by_game.items()}
    if not population or any(count <= 0 for count in population.values()):
        raise ValueError("positive state-population counts are required for every game")
    expected = [str(state_hash) for state_hash in expected_state_hashes]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected selected state hashes must be unique and non-empty")

    by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        state_hash = str(row.get("state_hash", ""))
        game_id = str(row.get("game_id", ""))
        if not state_hash or not game_id:
            raise ValueError("each Position row needs state_hash and game_id")
        student_branch = row.get("student_branch")
        teacher_branch = row.get("teacher_branch")
        if not isinstance(student_branch, Mapping) or not isinstance(
            teacher_branch, Mapping
        ):
            raise ValueError("each Position row needs both paired branch outcomes")
        if not isinstance(student_branch.get("won"), bool) or not isinstance(
            teacher_branch.get("won"), bool
        ):
            raise ValueError("paired branch won fields must be booleans")
        student_won = int(student_branch["won"])
        teacher_won = int(teacher_branch["won"])
        if int(row.get("consequence")) != teacher_won - student_won:
            raise ValueError("Position consequence disagrees with its paired outcomes")
        student_action = row.get("student_action")
        teacher_action = row.get("teacher_action")
        if not isinstance(student_action, str) or not isinstance(teacher_action, str):
            raise ValueError("Position rows must retain both intervention actions")
        expected_disagreement = int(student_action != teacher_action)
        if int(row.get("disagreement", -1)) != expected_disagreement:
            raise ValueError("Position D must equal Student/Teacher action inequality")
        if game_id not in population:
            raise ValueError(f"Position game {game_id!r} is outside the state population")
        by_state[state_hash].append(row)

    state_summaries: list[dict[str, Any]] = []
    invalid_probability_states: list[str] = []
    for state_hash in sorted(by_state):
        group = by_state[state_hash]
        games = {str(row["game_id"]) for row in group}
        probabilities = {row.get("inclusion_probability") for row in group}
        if len(games) != 1 or len(probabilities) != 1:
            raise ValueError("Position draws for one state disagree on game or design probability")
        probability_raw = next(iter(probabilities))
        probability = (
            None if probability_raw is None else float(probability_raw)
        )
        if probability is None or not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            invalid_probability_states.append(state_hash)
            probability = None
        student_mean = sum(
            float(bool(row["student_branch"]["won"])) for row in group
        ) / len(group)
        teacher_mean = sum(
            float(bool(row["teacher_branch"]["won"])) for row in group
        ) / len(group)
        consequence_mean = sum(float(row["consequence"]) for row in group) / len(group)
        divergent_draws = sum(int(row["disagreement"]) for row in group)
        divergence_mass = divergent_draws / len(group)
        divergent_consequence_mass = sum(
            float(row["consequence"]) * int(row["disagreement"]) for row in group
        ) / len(group)
        divergent_student_won_mass = sum(
            float(bool(row["student_branch"]["won"])) * int(row["disagreement"])
            for row in group
        ) / len(group)
        divergent_teacher_won_mass = sum(
            float(bool(row["teacher_branch"]["won"])) * int(row["disagreement"])
            for row in group
        ) / len(group)
        state_summaries.append(
            {
                "state_hash": state_hash,
                "game_id": next(iter(games)),
                "valid_teacher_draws": len(group),
                "divergent_teacher_draws": divergent_draws,
                "inclusion_probability": probability,
                "state_weight": None if probability is None else 1.0 / probability,
                "student_won": student_mean,
                "teacher_won": teacher_mean,
                "consequence": consequence_mean,
                "divergence_mass": divergence_mass,
                "divergent_consequence_mass": divergent_consequence_mass,
                "divergent_student_won_mass": divergent_student_won_mass,
                "divergent_teacher_won_mass": divergent_teacher_won_mass,
            }
        )

    sampled_by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in state_summaries:
        sampled_by_game[state["game_id"]].append(state)
    descriptive = None
    descriptive_given_disagreement = None
    if sampled_by_game:
        descriptive_by_game = {
            game: {
                outcome: sum(state[outcome] for state in states) / len(states)
                for outcome in ("student_won", "teacher_won", "consequence")
            }
            for game, states in sorted(sampled_by_game.items())
        }
        descriptive = {
            outcome: sum(values[outcome] for values in descriptive_by_game.values())
            / len(descriptive_by_game)
            for outcome in ("student_won", "teacher_won", "consequence")
        }
        descriptive["games"] = len(descriptive_by_game)
        descriptive["states"] = len(state_summaries)
        descriptive["estimand"] = "observed_replayable_sample_game_balanced_mean"
        descriptive_d_by_game = {
            game: {
                outcome: sum(state[outcome] for state in states) / len(states)
                for outcome in (
                    "divergence_mass",
                    "divergent_consequence_mass",
                    "divergent_student_won_mass",
                    "divergent_teacher_won_mass",
                )
            }
            for game, states in sorted(sampled_by_game.items())
        }
        d_denominator = sum(
            value["divergence_mass"] for value in descriptive_d_by_game.values()
        ) / len(descriptive_d_by_game)
        if d_denominator > 0.0:
            descriptive_given_disagreement = {
                "consequence": sum(
                    value["divergent_consequence_mass"]
                    for value in descriptive_d_by_game.values()
                )
                / len(descriptive_d_by_game)
                / d_denominator,
                "student_won": sum(
                    value["divergent_student_won_mass"]
                    for value in descriptive_d_by_game.values()
                )
                / len(descriptive_d_by_game)
                / d_denominator,
                "teacher_won": sum(
                    value["divergent_teacher_won_mass"]
                    for value in descriptive_d_by_game.values()
                )
                / len(descriptive_d_by_game)
                / d_denominator,
                "disagreement_mass": d_denominator,
                "games": len(descriptive_d_by_game),
                "states": len(state_summaries),
                "estimand": (
                    "observed_replayable_sample_game_balanced_mean_given_"
                    "D1_and_teacher_valid"
                ),
            }

    reasons: list[str] = []
    observed_hashes = set(by_state)
    expected_hash_set = set(expected)
    if observed_hashes != expected_hash_set:
        missing = len(expected_hash_set - observed_hashes)
        extra = len(observed_hashes - expected_hash_set)
        reasons.append(
            f"replayable state set differs from selected corrections (missing={missing}, extra={extra})"
        )
    if invalid_probability_states:
        reasons.append(
            "valid inclusion_probability is missing for "
            f"{len(invalid_probability_states)} replayed state(s)"
        )
    if set(sampled_by_game) != set(population):
        reasons.append("the selected replayable states do not cover every population game")

    population_estimate = None
    bootstrap = None
    bootstrap_nonidentification = None
    primary_population_estimate = None
    primary_bootstrap = None
    primary_nonidentification = list(reasons)
    primary_bootstrap_nonidentification = None
    if not reasons:
        per_game: dict[str, dict[str, float]] = {}
        for game in sorted(population):
            states = sampled_by_game[game]
            population_size = population[game]
            per_game[game] = {
                outcome: sum(
                    state[outcome] / float(state["inclusion_probability"])
                    for state in states
                )
                / population_size
                for outcome in (
                    "student_won",
                    "teacher_won",
                    "consequence",
                    "divergence_mass",
                    "divergent_consequence_mass",
                    "divergent_student_won_mass",
                    "divergent_teacher_won_mass",
                )
            }
        population_estimate = {
            outcome: sum(values[outcome] for values in per_game.values()) / len(per_game)
            for outcome in ("student_won", "teacher_won", "consequence")
        }
        population_estimate.update(
            {
                "games": len(per_game),
                "sampled_states": len(state_summaries),
                "estimand": "finite_state_population_game_balanced_ht_mean",
                "per_game": per_game,
            }
        )
        games = sorted(per_game)
        d_denominator = sum(
            per_game[game]["divergence_mass"] for game in games
        ) / len(games)
        if d_denominator <= 0.0:
            primary_nonidentification.append(
                "no Teacher-valid replay has D=1, so the conditional Position "
                "estimand has a zero denominator"
            )
        else:
            primary_population_estimate = {
                "consequence": sum(
                    per_game[game]["divergent_consequence_mass"] for game in games
                )
                / len(games)
                / d_denominator,
                "student_won": sum(
                    per_game[game]["divergent_student_won_mass"] for game in games
                )
                / len(games)
                / d_denominator,
                "teacher_won": sum(
                    per_game[game]["divergent_teacher_won_mass"] for game in games
                )
                / len(games)
                / d_denominator,
                "disagreement_mass": d_denominator,
                "games": len(games),
                "sampled_states": len(state_summaries),
                "estimand": (
                    "finite_state_population_game_balanced_ht_ratio_"
                    "E_C_given_D1_and_teacher_valid"
                ),
                "per_game_ht_components": {
                    game: {
                        "numerator": per_game[game]["divergent_consequence_mass"],
                        "denominator": per_game[game]["divergence_mass"],
                    }
                    for game in games
                },
            }

        if len(games) < 2:
            bootstrap_nonidentification = (
                "at least two game clusters are required for a non-degenerate "
                "game bootstrap"
            )
            primary_bootstrap_nonidentification = bootstrap_nonidentification
        else:
            rng = random.Random(bootstrap_seed)
            overall_draws: list[float] = []
            conditional_draws: list[float] = []
            attempts = 0
            while (
                len(overall_draws) < bootstrap_replicates
                or (
                    primary_population_estimate is not None
                    and len(conditional_draws) < bootstrap_replicates
                )
            ) and attempts < bootstrap_replicates * 20:
                attempts += 1
                sampled_games = [rng.choice(games) for _ in games]
                if len(overall_draws) < bootstrap_replicates:
                    overall_draws.append(
                        sum(per_game[game]["consequence"] for game in sampled_games)
                        / len(sampled_games)
                    )
                if (
                    primary_population_estimate is not None
                    and len(conditional_draws) < bootstrap_replicates
                ):
                    denominator = sum(
                        per_game[game]["divergence_mass"] for game in sampled_games
                    )
                    if denominator > 0.0:
                        conditional_draws.append(
                            sum(
                                per_game[game]["divergent_consequence_mass"]
                                for game in sampled_games
                            )
                            / denominator
                        )
            overall_draws.sort()
            alpha = 1.0 - confidence
            bootstrap = {
                "estimate": population_estimate["consequence"],
                "lower": _quantile(overall_draws, alpha / 2.0),
                "upper": _quantile(overall_draws, 1.0 - alpha / 2.0),
                "replicates": bootstrap_replicates,
                "confidence": confidence,
                "rng_seed": bootstrap_seed,
                "resampling_unit": "paired_game_cluster",
                "conditional_on_selected_state_design": True,
            }
            if primary_population_estimate is not None:
                if len(conditional_draws) != bootstrap_replicates:
                    primary_bootstrap_nonidentification = (
                        "D=1 is too sparse for the requested conditional game bootstrap"
                    )
                else:
                    conditional_draws.sort()
                    primary_bootstrap = {
                        "estimate": primary_population_estimate["consequence"],
                        "lower": _quantile(conditional_draws, alpha / 2.0),
                        "upper": _quantile(
                            conditional_draws, 1.0 - alpha / 2.0
                        ),
                        "replicates": bootstrap_replicates,
                        "confidence": confidence,
                        "rng_seed": bootstrap_seed,
                        "resampling_unit": "paired_game_cluster",
                        "conditional_on_selected_state_design": True,
                    }

    return {
        "protocol_version": "omniopd-v1",
        "population_states_by_game": dict(sorted(population.items())),
        "state_aggregation": "mean_over_valid_teacher_draws_before_design_weighting",
        "population_weighting": (
            "equal_game_mean_of_within_game_horvitz_thompson_state_means"
        ),
        "state_summaries": state_summaries,
        "descriptive_sample_game_balanced": descriptive,
        "descriptive_sample_game_balanced_given_disagreement": (
            descriptive_given_disagreement
        ),
        "population_identified": not reasons,
        "nonidentification_reasons": reasons,
        "population_game_balanced_ht": population_estimate,
        "paired_game_bootstrap": bootstrap,
        "paired_game_bootstrap_nonidentification": bootstrap_nonidentification,
        "primary_estimand": "E[C|D=1,V_T=1]",
        "primary_population_identified": (
            not primary_nonidentification and primary_population_estimate is not None
        ),
        "primary_nonidentification_reasons": primary_nonidentification,
        "population_game_balanced_ht_given_disagreement": (
            primary_population_estimate
        ),
        "paired_game_bootstrap_given_disagreement": primary_bootstrap,
        "paired_game_bootstrap_given_disagreement_nonidentification": (
            primary_bootstrap_nonidentification
        ),
    }


def _replay_to_state(
    env_factory: Callable[[], AgentEnvironment],
    prefix_actions: Sequence[str],
    expected: AgentState,
) -> tuple[AgentEnvironment, ResetResult]:
    env = env_factory()
    reset = env.reset()
    if reset.game_id != expected.game_id or reset.task_type != expected.task_type:
        env.close()
        raise RuntimeError("state-fidelity gate failed: reset game identity mismatch")
    if reset.task != expected.task:
        env.close()
        raise RuntimeError("state-fidelity gate failed: task mismatch")
    current_observation = reset.observation
    current_actions = reset.admissible_actions
    for action in prefix_actions:
        transition = env.step(action)
        if transition.done:
            env.close()
            raise RuntimeError("prefix terminated before the intervention state")
        current_observation = transition.observation
        current_actions = transition.admissible_actions
    if current_observation != expected.observation:
        env.close()
        raise RuntimeError("state-fidelity gate failed: observation mismatch")
    if tuple(current_actions) != expected.admissible_actions:
        env.close()
        raise RuntimeError("state-fidelity gate failed: ordered admissible actions mismatch")
    return env, reset


def _run_branch(
    env_factory: Callable[[], AgentEnvironment],
    prefix_actions: Sequence[str],
    state: AgentState,
    intervention_action: str,
    policy: ChatPolicy,
    truncator: TaskPreservingTruncator,
    settings: GenerationSettings,
    max_steps: int,
    request_namespace: str,
    branch_label: str,
    student_system_prompt: str = STUDENT_SYSTEM_PROMPT,
) -> BranchOutcome:
    if intervention_action not in state.admissible_actions:
        raise ValueError("intervention action is not admissible")
    env, _ = _replay_to_state(env_factory, prefix_actions, state)
    # Reconstruct the continuation from the full executed history. Reusing the
    # already-truncated query would permanently discard older pairs and can
    # change what a fresh policy query retains at later branch steps. The
    # continuation policy is always the Student, including for Teacher-state
    # source controls, so its system prompt must be P_S -- passed in by the
    # caller, because the frozen P_S of a study need not be the repository
    # default (a mismatch here silently changes the continuation policy).
    full_history = state.full_messages or state.messages
    history = ConversationHistory(
        task=state.task,
        game_id=state.game_id,
        task_type=state.task_type,
        _messages=replace_system(full_history, student_system_prompt),
        _observation=state.observation,
        _admissible_actions=state.admissible_actions,
        _turn_index=state.turn_index,
    )
    executed = [intervention_action]
    try:
        history.record_action(intervention_action)
        transition = env.step(intervention_action)
        history.record_observation(
            transition.observation, transition.admissible_actions, done=transition.done
        )
        done_after = transition.done
        steps = 0
        while not transition.done and len(prefix_actions) + 1 + steps < max_steps:
            history.assert_query_ready()
            query, _ = truncator.truncate(history.messages)
            raw = policy.generate(
                query,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                request_id=f"{request_namespace}:{branch_label}:{steps}",
            )
            parsed = parse_action(
                raw,
                transition.admissible_actions,
                allow_unique_prefix=settings.allow_unique_prefix,
            )
            action = parsed.canonical_action if parsed.valid else settings.fallback_action
            if action not in transition.admissible_actions:
                raise RuntimeError("counterfactual fallback is not admissible")
            executed.append(action)
            history.record_action(action)
            transition = env.step(action)
            history.record_observation(
                transition.observation, transition.admissible_actions, done=transition.done
            )
            steps += 1
        return BranchOutcome(
            intervention_action,
            transition.won,
            done_after,
            steps,
            tuple(executed),
        )
    finally:
        env.close()


def run_paired_counterfactual(
    *,
    env_factory: Callable[[], AgentEnvironment],
    prefix_actions: Sequence[str],
    state: AgentState,
    student_action: str,
    teacher_action: str,
    frozen_student: ChatPolicy,
    truncator: TaskPreservingTruncator,
    settings: GenerationSettings = GenerationSettings(),
    max_steps: int = 50,
    request_namespace: str | None = None,
    student_system_prompt: str = STUDENT_SYSTEM_PROMPT,
) -> CounterfactualPair:
    """Replay two independent branches; the intervention action is the only change."""

    if max_steps <= 0 or len(prefix_actions) >= max_steps:
        raise ValueError("counterfactual intervention must occur within max_steps")
    if len(prefix_actions) != state.turn_index:
        raise ValueError("prefix length must equal the intervention state's turn index")
    namespace = request_namespace or f"cf:{state.state_hash}"
    if not namespace.strip() or any(character in namespace for character in "\r\n"):
        raise ValueError("request_namespace must be a non-empty single-line string")
    if state.full_messages:
        expected_length = 2 + 2 * state.turn_index
        if len(state.full_messages) != expected_length:
            raise ValueError("full state history length is inconsistent with turn_index")
        recorded_actions = []
        for message in state.full_messages[2::2]:
            content = message.get("content", "")
            if message.get("role") != "assistant" or not content.startswith("Action: "):
                raise ValueError("full state history contains a malformed executed action")
            recorded_actions.append(content[len("Action: ") :])
        if list(prefix_actions) != recorded_actions:
            raise ValueError("replay prefix does not match the recorded executed history")

    student = _run_branch(
        env_factory,
        prefix_actions,
        state,
        student_action,
        frozen_student,
        truncator,
        settings,
        max_steps,
        namespace,
        "student_action_branch",
        student_system_prompt,
    )
    teacher = _run_branch(
        env_factory,
        prefix_actions,
        state,
        teacher_action,
        frozen_student,
        truncator,
        settings,
        max_steps,
        namespace,
        "teacher_action_branch",
        student_system_prompt,
    )
    return CounterfactualPair(student, teacher)
