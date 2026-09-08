from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ..prompts import STUDENT_SYSTEM_PROMPT, replace_system
from ..provenance import sha256_json
from ..schema import RolloutTurn
from .transfer import bag_of_words, bow_cosine, set_jaccard

M3_GROUPS = frozenset({"A1", "A3"})
M3_STOPWORDS = frozenset(
    {"the", "a", "an", "you", "are", "is", "in", "of", "to", "and", "on", "with", "at", "from"}
)


def _source_id(
    group: str,
    experiment: str,
    state_hash: str,
    teacher_sample_index: int,
) -> str:
    return sha256_json(
        {
            "kind": "m3_source_v1",
            "group": group,
            "experiment": experiment,
            "state_hash": state_hash,
            "teacher_sample_index": teacher_sample_index,
        }
    )


def _score_id(source_id: str, target_kind: str, target_state_hash: str) -> str:
    return sha256_json(
        {
            "kind": "m3_score_v1",
            "source_id": source_id,
            "target_kind": target_kind,
            "target_state_hash": target_state_hash,
        }
    )


def _score_messages(turn: RolloutTurn, teacher_action: str) -> list[dict[str, str]]:
    messages = replace_system(turn.state.messages, STUDENT_SYSTEM_PROMPT)
    messages.append({"role": "assistant", "content": f"Action: {teacher_action}"})
    return messages


def _similarity(
    source: RolloutTurn,
    target: RolloutTurn,
    teacher_action: str,
    *,
    bow_weight: float,
) -> float:
    observation_similarity = bow_cosine(
        bag_of_words(source.state.observation, M3_STOPWORDS),
        bag_of_words(target.state.observation, M3_STOPWORDS),
    )
    source_actions = set(source.state.admissible_actions) - {teacher_action}
    target_actions = set(target.state.admissible_actions) - {teacher_action}
    action_similarity = set_jaccard(source_actions, target_actions)
    return bow_weight * observation_similarity + (1.0 - bow_weight) * (
        0.0 if action_similarity is None else action_similarity
    )


def build_m3_panel(
    turns: Sequence[RolloutTurn],
    source_rows_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_hashes_by_design: Mapping[str, set[str]],
    *,
    experiments_by_group: Mapping[str, str],
    neighbors: int = 10,
    minimum_neighbors: int = 5,
    minimum_similarity: float = 0.0,
    bow_weight: float = 0.5,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze M3 source/neighbor identities before either checkpoint is scored.

    The transfer pool excludes every state named by any supplied selection
    design. Source actions are only transferred to other games where the exact
    action is admissible. Neighbors are selected once, deterministically, and
    then reused by the base and updated checkpoints.
    """

    if set(source_rows_by_group) != M3_GROUPS:
        raise ValueError("M3 source groups must be exactly A1 and A3")
    if set(experiments_by_group) != M3_GROUPS:
        raise ValueError("M3 experiment identities must be supplied for A1 and A3")
    normalized_experiments = {
        group: str(experiments_by_group[group]).strip() for group in M3_GROUPS
    }
    if any(not value for value in normalized_experiments.values()) or len(
        set(normalized_experiments.values())
    ) != len(M3_GROUPS):
        raise ValueError("M3 A1/A3 experiment identities must be non-empty and distinct")
    if not M3_GROUPS <= set(selected_hashes_by_design):
        raise ValueError("M3 requires explicit A1 and A3 selection sets")
    if neighbors <= 0 or not 1 <= minimum_neighbors <= neighbors:
        raise ValueError("M3 neighbor counts must satisfy 1 <= minimum <= K")
    if not 0.0 <= minimum_similarity <= 1.0 or not 0.0 <= bow_weight <= 1.0:
        raise ValueError("M3 similarity thresholds and weights must lie in [0,1]")

    by_hash = {turn.state.state_hash: turn for turn in turns}
    if not turns or len(by_hash) != len(turns):
        raise ValueError("M3 state pool must be non-empty with unique state hashes")
    excluded = set().union(*selected_hashes_by_design.values())
    unknown_selected = sorted(excluded - set(by_hash))
    if unknown_selected:
        raise ValueError(f"M3 selections contain states outside the frozen pool: {unknown_selected[:5]}")

    by_game: dict[str, list[RolloutTurn]] = defaultdict(list)
    for turn in turns:
        by_game[turn.state.game_id].append(turn)
    previous_observation: dict[str, str | None] = {}
    for game_turns in by_game.values():
        previous = None
        seen_turn_indices: set[int] = set()
        for turn in sorted(game_turns, key=lambda item: item.state.turn_index):
            if turn.state.turn_index in seen_turn_indices:
                raise ValueError("M3 state pool contains duplicate game/turn identities")
            seen_turn_indices.add(turn.state.turn_index)
            previous_observation[turn.state.state_hash] = previous
            previous = turn.state.observation

    transfer_pool = [
        turn
        for turn in turns
        if turn.student.valid and turn.state.state_hash not in excluded
    ]
    score_rows: dict[str, list[dict[str, Any]]] = {group: [] for group in sorted(M3_GROUPS)}
    metadata: list[dict[str, Any]] = []
    attrition: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for group in sorted(M3_GROUPS):
        panel_experiment = normalized_experiments[group]
        selection = selected_hashes_by_design[group]
        ordered_sources = sorted(
            source_rows_by_group[group],
            key=lambda row: (
                str(row.get("state_hash", "")),
                int(row.get("teacher_sample_index", -1)),
            ),
        )
        if not ordered_sources:
            raise ValueError(f"M3 source group {group} is empty")
        for row in ordered_sources:
            required = {
                "state_hash",
                "game_id",
                "task_type",
                "turn_index",
                "teacher_sample_index",
                "teacher_action",
                "student_action",
                "state_source",
                "protocol_version",
                "enable_thinking",
            }
            missing = required - set(row)
            if missing:
                raise ValueError(f"M3 source row is missing fields: {sorted(missing)}")
            state_hash = str(row["state_hash"])
            if state_hash not in by_hash or state_hash not in selection:
                raise ValueError("every M3 source must belong to its group selection and state pool")
            source = by_hash[state_hash]
            if (
                str(row["game_id"]) != source.state.game_id
                or str(row["task_type"]) != source.state.task_type
                or int(row["turn_index"]) != source.state.turn_index
                or str(row["state_source"]) != source.state.state_source
                or str(row["student_action"]) != source.student.executed_action
            ):
                raise ValueError("M3 source metadata disagrees with the frozen state pool")
            if row["protocol_version"] != "omniopd-v1" or bool(row["enable_thinking"]):
                raise ValueError("M3 sources must use the canonical non-thinking protocol")
            sample_index = int(row["teacher_sample_index"])
            teacher_action = str(row["teacher_action"])
            if sample_index < 0 or not teacher_action.strip():
                raise ValueError("M3 Teacher sample identity/action is invalid")
            if teacher_action not in source.state.admissible_actions:
                raise ValueError("M3 source Teacher action is not admissible at its source state")
            source_id = _source_id(
                group, panel_experiment, state_hash, sample_index
            )
            if source_id in seen_sources:
                raise ValueError("M3 contains a duplicate group/state/Teacher-sample source")
            seen_sources.add(source_id)

            candidates = []
            for target in transfer_pool:
                if (
                    target.state.task_type != source.state.task_type
                    or target.state.game_id == source.state.game_id
                    or teacher_action not in target.state.admissible_actions
                ):
                    continue
                similarity = _similarity(
                    source, target, teacher_action, bow_weight=bow_weight
                )
                if similarity >= minimum_similarity:
                    candidates.append((similarity, target))
            candidates.sort(key=lambda item: (-item[0], item[1].state.state_hash))
            chosen = candidates[:neighbors]
            if len(chosen) < minimum_neighbors:
                attrition.append(
                    {
                        "source_id": source_id,
                        "group": group,
                        "source_state_hash": state_hash,
                        "reason": "insufficient_eligible_cross_game_neighbors",
                        "eligible_neighbors": len(candidates),
                        "minimum_neighbors": minimum_neighbors,
                    }
                )
                continue

            self_score_id = _score_id(source_id, "self", state_hash)
            score_rows[group].append(
                _make_score_row(
                    score_id=self_score_id,
                    source_id=source_id,
                    group=group,
                    panel_experiment=panel_experiment,
                    target_kind="self",
                    source_state_hash=state_hash,
                    target=source,
                    teacher_sample_index=sample_index,
                    teacher_action=teacher_action,
                )
            )
            neighbor_rows = []
            for similarity, target in chosen:
                score_id = _score_id(
                    source_id, "neighbor", target.state.state_hash
                )
                score_rows[group].append(
                    _make_score_row(
                        score_id=score_id,
                        source_id=source_id,
                        group=group,
                        panel_experiment=panel_experiment,
                        target_kind="neighbor",
                        source_state_hash=state_hash,
                        target=target,
                        teacher_sample_index=sample_index,
                        teacher_action=teacher_action,
                    )
                )
                neighbor_rows.append(
                    {
                        "score_id": score_id,
                        "state_hash": target.state.state_hash,
                        "game_id": target.state.game_id,
                        "turn_index": target.state.turn_index,
                        "similarity": similarity,
                    }
                )
            student_words = str(row["student_action"]).split()
            previous = previous_observation[state_hash]
            metadata.append(
                {
                    "protocol_version": "omniopd-v1",
                    "source_id": source_id,
                    "group": group,
                    "panel_group": group,
                    "panel_experiment": panel_experiment,
                    "source_score_id": self_score_id,
                    "source_state_hash": state_hash,
                    "game_id": source.state.game_id,
                    "task_type": source.state.task_type,
                    "turn_index": source.state.turn_index,
                    "teacher_sample_index": sample_index,
                    "teacher_action": teacher_action,
                    "student_action": str(row["student_action"]),
                    "action_type": student_words[0].lower() if student_words else "none",
                    "technical": int(not source.student.valid),
                    "repeat_obs": int(
                        previous is not None
                        and bool(source.state.observation)
                        and source.state.observation == previous
                    ),
                    "sim_topK": sum(value for value, _ in chosen) / len(chosen),
                    "n_neighbors": len(chosen),
                    "neighbor_set_sha256": sha256_json(neighbor_rows),
                    "neighbors": neighbor_rows,
                }
            )
    # The panel comparison must target the same game support. Teacher-valid
    # filtering and neighbor attrition can otherwise leave A1 and A3 defined on
    # different games, confounding the panel effect with task composition.
    retained_games = {
        group: {
            str(row["game_id"])
            for row in metadata
            if row["group"] == group
        }
        for group in M3_GROUPS
    }
    common_games = set.intersection(*retained_games.values())
    dropped_metadata = [
        row for row in metadata if str(row["game_id"]) not in common_games
    ]
    for row in dropped_metadata:
        attrition.append(
            {
                "source_id": row["source_id"],
                "group": row["group"],
                "source_state_hash": row["source_state_hash"],
                "reason": "outside_cross_panel_common_game_support",
                "source_game": row["game_id"],
            }
        )
    retained_source_ids = {
        str(row["source_id"])
        for row in metadata
        if str(row["game_id"]) in common_games
    }
    metadata = [
        row for row in metadata if str(row["source_id"]) in retained_source_ids
    ]
    for group in score_rows:
        score_rows[group] = [
            row
            for row in score_rows[group]
            if str(row["source_id"]) in retained_source_ids
        ]
        score_rows[group].sort(key=lambda row: row["score_id"])
    metadata.sort(key=lambda row: row["source_id"])
    attrition.sort(key=lambda row: row["source_id"])
    return score_rows, metadata, attrition


def _make_score_row(
    *,
    score_id: str,
    source_id: str,
    group: str,
    panel_experiment: str,
    target_kind: str,
    source_state_hash: str,
    target: RolloutTurn,
    teacher_sample_index: int,
    teacher_action: str,
) -> dict[str, Any]:
    return {
        "protocol_version": "omniopd-v1",
        "score_id": score_id,
        "source_id": source_id,
        "group": group,
        "panel_group": group,
        "panel_experiment": panel_experiment,
        "target_kind": target_kind,
        "source_state_hash": source_state_hash,
        "target_state_hash": target.state.state_hash,
        "state_hash": target.state.state_hash,
        "game_id": target.state.game_id,
        "task_type": target.state.task_type,
        "turn_index": target.state.turn_index,
        "teacher_sample_index": teacher_sample_index,
        "teacher_action": teacher_action,
        "messages": _score_messages(target, teacher_action),
        "enable_thinking": False,
    }


def assemble_m3_rows(
    source_metadata: Sequence[Mapping[str, Any]],
    base_scores_by_panel: Mapping[str, Sequence[Mapping[str, Any]]],
    updated_scores_by_cell: Mapping[
        tuple[str, str], Sequence[Mapping[str, Any]]
    ],
) -> list[dict[str, Any]]:
    """Join the required checkpoint × panel 2×2 M3 scoring design.

    Each A1/A3 source panel is scored by both the A1 and A3 final checkpoints,
    against one shared base score for that panel.  This prevents checkpoint
    identity from being perfectly confounded with source-panel identity.
    """

    if set(base_scores_by_panel) != M3_GROUPS:
        raise ValueError("M3 base scores must cover the A1 and A3 panels")
    expected_cells = {
        (checkpoint, panel)
        for checkpoint in M3_GROUPS
        for panel in M3_GROUPS
    }
    if set(updated_scores_by_cell) != expected_cells:
        raise ValueError("M3 updated scores must form a complete checkpoint×panel 2x2 grid")
    if not source_metadata:
        raise ValueError("M3 source metadata is empty")
    metadata_ids = [str(row["source_id"]) for row in source_metadata]
    if len(metadata_ids) != len(set(metadata_ids)):
        raise ValueError("M3 source metadata contains duplicate source IDs")
    if {
        str(row.get("panel_group", row.get("group"))) for row in source_metadata
    } != M3_GROUPS:
        raise ValueError("M3 source metadata must retain separate A1 and A3 rows")
    experiments_by_group: dict[str, str] = {}
    for row in source_metadata:
        panel = str(row.get("panel_group", row.get("group")))
        experiment = str(row.get("panel_experiment", "")).strip()
        if panel not in M3_GROUPS or not experiment:
            raise ValueError("M3 source metadata lacks a panel experiment identity")
        previous = experiments_by_group.setdefault(panel, experiment)
        if previous != experiment:
            raise ValueError("one M3 panel is bound to multiple experiment identities")
    if set(experiments_by_group) != M3_GROUPS or len(
        set(experiments_by_group.values())
    ) != 2:
        raise ValueError("M3 A1/A3 panel experiments must be distinct")

    assembled = []
    for panel in sorted(M3_GROUPS):
        group_metadata = [
            row
            for row in source_metadata
            if row.get("panel_group", row.get("group")) == panel
        ]
        expected_ids = {
            str(row["source_score_id"])
            for row in group_metadata
        } | {
            str(neighbor["score_id"])
            for row in group_metadata
            for neighbor in row["neighbors"]
        }
        base = _index_scores(
            base_scores_by_panel[panel],
            panel_group=panel,
            checkpoint_group="BASE",
            checkpoint_experiment=None,
        )
        if set(base) != expected_ids:
            raise ValueError(
                f"M3 BASE×{panel} scores do not exactly cover the frozen panel"
            )
        for checkpoint in sorted(M3_GROUPS):
            updated = _index_scores(
                updated_scores_by_cell[(checkpoint, panel)],
                panel_group=panel,
                checkpoint_group=checkpoint,
                checkpoint_experiment=experiments_by_group[checkpoint],
            )
            if set(updated) != expected_ids:
                raise ValueError(
                    f"M3 {checkpoint}×{panel} scores do not exactly cover the frozen panel"
                )
            for score_id in expected_ids:
                identity_fields = [
                    "score_id",
                    "source_id",
                    "group",
                    "panel_group",
                    "panel_experiment",
                    "target_kind",
                    "source_state_hash",
                    "target_state_hash",
                    "teacher_action",
                    "target_tokens",
                ]
                if any(
                    base[score_id].get(key) != updated[score_id].get(key)
                    for key in identity_fields
                ):
                    raise ValueError("M3 base and updated scores refer to different examples")
            for metadata in group_metadata:
                source_score_id = str(metadata["source_score_id"])
                source_base = float(
                    base[source_score_id]["mean_target_log_probability"]
                )
                source_updated = float(
                    updated[source_score_id]["mean_target_log_probability"]
                )
                neighbor_ids = [
                    str(row["score_id"]) for row in metadata["neighbors"]
                ]
                neighbor_base = [
                    float(base[score_id]["mean_target_log_probability"])
                    for score_id in neighbor_ids
                ]
                neighbor_updated = [
                    float(updated[score_id]["mean_target_log_probability"])
                    for score_id in neighbor_ids
                ]
                neighbor_deltas = [
                    after - before
                    for before, after in zip(neighbor_base, neighbor_updated)
                ]
                row = dict(metadata)
                row["source_group"] = row.pop("group", panel)
                row.update(
                    {
                        "panel_group": panel,
                        "checkpoint_group": checkpoint,
                        "checkpoint_experiment": experiments_by_group[checkpoint],
                        "S_i": source_updated - source_base,
                        "T_i": sum(neighbor_deltas) / len(neighbor_deltas),
                        "source_base_logp": source_base,
                        "source_updated_logp": source_updated,
                        "neighbor_base_logp_mean": sum(neighbor_base)
                        / len(neighbor_base),
                        "neighbor_updated_logp_mean": sum(neighbor_updated)
                        / len(neighbor_updated),
                        "neighbor_dlogp": neighbor_deltas,
                        "estimand": "crossed_whole_checkpoint_action_imitation_transfer_not_task_utility",
                    }
                )
                assembled.append(row)
    assembled.sort(key=lambda row: (row["source_id"], row["checkpoint_group"]))
    return assembled


def _index_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    panel_group: str,
    checkpoint_group: str,
    checkpoint_experiment: str | None,
) -> dict[str, Mapping[str, Any]]:
    required = {
        "score_id",
        "source_id",
        "group",
        "panel_group",
        "panel_experiment",
        "checkpoint_group",
        "checkpoint_experiment",
        "target_kind",
        "source_state_hash",
        "target_state_hash",
        "teacher_action",
        "mean_target_log_probability",
        "target_tokens",
    }
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"M3 score row is missing fields: {sorted(missing)}")
        score_id = str(row["score_id"])
        value = float(row["mean_target_log_probability"])
        if (
            row["group"] != panel_group
            or row["panel_group"] != panel_group
            or row["checkpoint_group"] != checkpoint_group
            or row["checkpoint_experiment"] != checkpoint_experiment
            or row["target_kind"] not in {"self", "neighbor"}
        ):
            raise ValueError(
                "M3 score row has the wrong panel/checkpoint experiment identity"
            )
        if score_id in indexed or not math.isfinite(value) or int(row["target_tokens"]) <= 0:
            raise ValueError("M3 scores must be unique, finite, and contain target tokens")
        indexed[score_id] = row
    return indexed
