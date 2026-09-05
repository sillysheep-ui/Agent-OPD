from __future__ import annotations

import random
import json
from collections import defaultdict
from typing import Any, Iterable, Sequence


def _row_id(row: dict[str, Any]) -> str:
    if row.get("sample_id") is not None:
        return str(row["sample_id"])
    state_hash = row.get("state_hash")
    if state_hash is None and isinstance(row.get("state"), dict):
        state_hash = row["state"].get("state_hash")
    if state_hash is not None:
        return f"{state_hash}:{row.get('teacher_sample_index', '')}"
    return json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)


def exact_stratified_sample(
    candidates: Iterable[dict[str, Any]],
    targets: Iterable[dict[str, Any]],
    *,
    strata: Sequence[str],
    rng_seed: int,
) -> list[dict[str, Any]]:
    """Match target counts within preregistered strata; fail on support violations."""

    candidates_by_stratum: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    target_counts: dict[tuple, int] = defaultdict(int)
    seen_candidates = set()
    for candidate in candidates:
        if any(key not in candidate for key in strata):
            raise ValueError("every candidate must contain every matching stratum")
        row_id = _row_id(candidate)
        if row_id in seen_candidates:
            raise ValueError(f"duplicate candidate identity: {row_id}")
        seen_candidates.add(row_id)
        candidates_by_stratum[tuple(candidate[key] for key in strata)].append(candidate)
    for target in targets:
        if any(key not in target for key in strata):
            raise ValueError("every target must contain every matching stratum")
        target_counts[tuple(target[key] for key in strata)] += 1
    rng = random.Random(rng_seed)
    selected = []
    for stratum, count in sorted(target_counts.items(), key=lambda item: repr(item[0])):
        pool = sorted(candidates_by_stratum.get(stratum, []), key=_row_id)
        if len(pool) < count:
            raise ValueError(
                f"insufficient candidate support for stratum {stratum}: need {count}, have {len(pool)}"
            )
        selected.extend(rng.sample(pool, count))
    return sorted(selected, key=_row_id)
