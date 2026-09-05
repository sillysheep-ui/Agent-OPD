from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class RegressionResult:
    coefficients: dict[str, float]
    rows: int
    games: int


def _game_id(row: dict[str, Any]) -> str:
    value = row.get("game_id", row.get("gamefile"))
    if value is None:
        raise ValueError("every M3 row requires game_id (or legacy gamefile)")
    return str(value)


def _design(rows: Sequence[dict[str, Any]], *, levels: dict[str, list[str]] | None = None):
    import numpy as np

    if not rows:
        raise ValueError("regression rows cannot be empty")
    levels = levels or {
        "task_type": sorted({str(row["task_type"]) for row in rows}),
        "action_type": sorted({str(row["action_type"]) for row in rows}),
    }
    names = ["intercept", "group_A3", "S", "S_x_group_A3", "sim", "technical", "repeat_obs"]
    names += [f"task_type[{value}]" for value in levels["task_type"][1:]]
    names += [f"action_type[{value}]" for value in levels["action_type"][1:]]
    matrix = []
    outcomes = []
    for row in rows:
        group = 1.0 if row["group"] == "A3" else 0.0
        source = float(row["S_i"])
        vector = [
            1.0,
            group,
            source,
            source * group,
            float(row["sim_topK"]),
            float(row.get("technical", 0)),
            float(row.get("repeat_obs", 0)),
        ]
        vector += [float(row["task_type"] == value) for value in levels["task_type"][1:]]
        vector += [float(row["action_type"] == value) for value in levels["action_type"][1:]]
        matrix.append(vector)
        outcomes.append(float(row["T_i"]))
    return np.asarray(matrix), np.asarray(outcomes), names, levels


def fit_transfer_regression(rows: Iterable[dict[str, Any]]) -> RegressionResult:
    import numpy as np

    rows = list(rows)
    groups = {row.get("group") for row in rows}
    if groups != {"A1", "A3"}:
        raise ValueError(f"M3 regression requires separate A1/A3 rows, got {groups}")
    matrix, outcomes, names, _ = _design(rows)
    if not np.isfinite(matrix).all() or not np.isfinite(outcomes).all():
        raise ValueError("M3 regression contains non-finite values")
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < matrix.shape[1]:
        raise ValueError(
            "M3 design matrix is rank deficient; named coefficients are not uniquely "
            f"identified (rank={rank}, columns={matrix.shape[1]})"
        )
    coefficients, *_ = np.linalg.lstsq(matrix, outcomes, rcond=None)
    return RegressionResult(
        dict(zip(names, coefficients.tolist())),
        len(rows),
        len({_game_id(row) for row in rows}),
    )


def game_cluster_bootstrap_coefficient(
    rows: Iterable[dict[str, Any]],
    *,
    coefficient: str,
    replicates: int = 5_000,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Cluster resample games while retaining A1/A3 as separate correction rows."""

    import numpy as np

    rows = list(rows)
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    games = sorted({_game_id(row) for row in rows})
    if len(games) < 2:
        raise ValueError("at least two game clusters are required")
    by_game = {game: [row for row in rows if _game_id(row) == game] for game in games}
    incomplete = [
        game
        for game, game_rows in by_game.items()
        if {row.get("group") for row in game_rows} != {"A1", "A3"}
    ]
    if incomplete:
        raise ValueError(
            "paired game-cluster bootstrap requires both A1 and A3 in every game; "
            f"missing in {incomplete[:5]}"
        )
    _, _, names, levels = _design(rows)
    if coefficient not in names:
        raise ValueError(f"unknown coefficient {coefficient!r}; available={names}")
    index = names.index(coefficient)
    point = fit_transfer_regression(rows).coefficients[coefficient]
    rng = random.Random(rng_seed)
    draws = []
    attempts = 0
    maximum_attempts = replicates * 20
    while len(draws) < replicates and attempts < maximum_attempts:
        attempts += 1
        sampled_rows = []
        for draw_index in range(len(games)):
            game = rng.choice(games)
            # Duplicated clusters are intentionally duplicated in the bootstrap sample.
            sampled_rows.extend(dict(row, _bootstrap_cluster=draw_index) for row in by_game[game])
        matrix, outcomes, _, _ = _design(sampled_rows, levels=levels)
        if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
            continue
        coefficients, *_ = np.linalg.lstsq(matrix, outcomes, rcond=None)
        draws.append(float(coefficients[index]))
    if len(draws) != replicates:
        raise ValueError(
            "too many rank-deficient game-cluster bootstrap draws; the requested "
            "coefficient is not stably identified by this design"
        )
    lower, upper = np.quantile(np.asarray(draws), [0.025, 0.975]).tolist()
    return point, lower, upper
