#!/usr/bin/env python3
"""Build audited fixed-pool token-OPD prompts; does not launch training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.io import read_jsonl, rollout_turn_from_dict
from omniopd.opd_adapter import build_fixed_pool_opd_prompts
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file
from omniopd.validation import (
    state_pool_behavior_student_contract,
    validate_selection_manifest_against_rows,
)


def _require_fields(actual: dict, expected: dict, *, role: str) -> None:
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{role} does not match this run: {mismatches}")


def build(
    *,
    state_pool_path: Path,
    state_pool_manifest_path: Path,
    selection_path: Path,
    selection_manifest_path: Path,
    experiment_path: Path,
    output: Path,
    manifest_output: Path,
) -> None:
    inputs = (
        state_pool_path,
        state_pool_manifest_path,
        selection_path,
        selection_manifest_path,
        experiment_path,
    )
    if any(not path.is_file() for path in inputs):
        raise ValueError("all state-pool, selection, and config inputs must be files")
    if (
        output == manifest_output
        or output in inputs
        or manifest_output in inputs
        or output.exists()
        or manifest_output.exists()
    ):
        raise ValueError("outputs must be new, distinct files outside the inputs")
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    if not isinstance(experiment, dict):
        raise ValueError("experiment config must be a mapping")
    try:
        policy = experiment["selection"]
        seed = experiment["selection_seed"]
        games = int(experiment["games"])
        states_per_game = int(experiment["states_per_game"])
        states = int(experiment["distinct_states_M"])
        experiment_name = str(experiment["experiment"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"experiment config is incomplete: {error}") from error
    if policy != "uniform_per_game_nested_v1":
        raise ValueError("this OPD prompt builder currently supports uniform nested selection only")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("uniform selection seed must be a non-negative integer")
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if not current_revision:
        raise ValueError("OPD prompt builder requires an OmniOPD Git revision")
    pool_sha = sha256_file(state_pool_path)
    pool_manifest_sha = sha256_file(state_pool_manifest_path)
    selection_sha = sha256_file(selection_path)
    pool_manifest = json.loads(state_pool_manifest_path.read_text(encoding="utf-8"))
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    _require_fields(
        pool_manifest,
        {
            "artifact": "immutable_state_pool",
            "protocol_version": "omniopd-v1",
            "code": current_code,
            "code_revision": current_revision,
        },
        role="state-pool manifest",
    )
    if pool_manifest.get("outputs", {}).get("state_pool_sha256") != pool_sha:
        raise ValueError("state-pool content does not match its manifest")
    behavior_student = state_pool_behavior_student_contract(pool_manifest)
    realized_state_pool = {
        "state_source": pool_manifest.get("state_source"),
        "games_G": pool_manifest.get("games_G"),
        "seed": pool_manifest.get("seed"),
        "environment_seed": pool_manifest.get("environment_rollout", {}).get("master_seed"),
        "max_steps": pool_manifest.get("max_steps"),
        "max_context_tokens": pool_manifest.get("max_context_tokens"),
        "reserve_tokens": pool_manifest.get("reserve_tokens"),
        "behavior_max_tokens": pool_manifest.get("behavior_max_tokens"),
        "behavior_sampling": pool_manifest.get("behavior_sampling"),
    }
    if realized_state_pool != experiment.get("state_pool"):
        raise ValueError("state-pool manifest disagrees with preregistered experiment config")
    _require_fields(
        selection_manifest,
        {
            "artifact": "selection_manifest",
            "protocol_version": "omniopd-v1",
            "code": current_code,
            "code_revision": current_revision,
            "experiment_config_sha256": sha256_file(experiment_path),
            "state_pool_sha256": pool_sha,
            "state_pool_manifest_sha256": pool_manifest_sha,
            "selection_sha256": selection_sha,
            "behavior_student": behavior_student,
            "policy": policy,
            "score_file_sha256": None,
            "score_manifest_sha256": None,
        },
        role="selection manifest",
    )
    turns = [rollout_turn_from_dict(row) for row in read_jsonl(state_pool_path)]
    selections = list(read_jsonl(selection_path))
    pool_hashes_by_game: dict[str, list[str]] = {}
    for turn in sorted(turns, key=lambda item: (item.state.game_id, item.state.turn_index)):
        pool_hashes_by_game.setdefault(turn.state.game_id, []).append(turn.state.state_hash)
    validate_selection_manifest_against_rows(
        selection_manifest,
        selections,
        pool_state_hashes_by_game=pool_hashes_by_game,
        expected_experiment=experiment_name,
        expected_policy=policy,
        expected_seed=seed,
        expected_games=games,
        expected_states_per_game=states_per_game,
        expected_states=states,
    )
    rows = build_fixed_pool_opd_prompts(turns, selections)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    manifest = {
        "artifact": "fixed_pool_token_opd_prompts",
        "schema_version": 1,
        "protocol_version": "omniopd-v1",
        "code": current_code,
        "code_revision": current_revision,
        "experiment": experiment_name,
        "experiment_config_sha256": sha256_file(experiment_path),
        "state_pool_sha256": pool_sha,
        "state_pool_manifest_sha256": pool_manifest_sha,
        "selection_sha256": selection_sha,
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "prompts_sha256": sha256_file(output),
        "states": len(rows),
        "games": games,
        "prompt_contract": "student_prompt_plus_separate_teacher_prompt_v1",
        "training_ready": False,
        "missing": [
            "Student sampled-token rollout with action-only mask",
            "Teacher scoring under distinct Teacher prompt",
            "matched game split and budget protocol",
        ],
    }
    with manifest_output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-pool", type=Path, required=True)
    parser.add_argument("--state-pool-manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        build(
            state_pool_path=args.state_pool.resolve(),
            state_pool_manifest_path=args.state_pool_manifest.resolve(),
            selection_path=args.selection.resolve(),
            selection_manifest_path=args.selection_manifest.resolve(),
            experiment_path=args.experiment_config.resolve(),
            output=args.output.resolve(),
            manifest_output=args.manifest_output.resolve(),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"OPD prompt build rejected: {error}") from error
    print(f"Audited OPD prompt rows written to {args.output}; training_ready=false")


if __name__ == "__main__":
    main()
