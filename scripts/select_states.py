#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.io import read_jsonl, rollout_turn_from_dict, write_jsonl
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file
from omniopd.selection import select_top_score, select_uniform_nested


def main() -> None:
    parser = argparse.ArgumentParser(description="Select states from one frozen rollout pool")
    parser.add_argument("--state-pool", required=True)
    parser.add_argument("--state-pool-manifest", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output")
    parser.add_argument(
        "--policy",
        choices=["uniform_per_game_nested_v1", "top_score_per_game"],
    )
    parser.add_argument("--states-per-game", type=int)
    parser.add_argument("--scores", help="JSONL containing state_hash and score")
    parser.add_argument("--scores-manifest")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    experiment_path = Path(args.experiment_config)
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    try:
        configured_policy = str(experiment["selection"])
        configured_states_per_game = int(experiment["states_per_game"])
        configured_states = int(experiment["distinct_states_M"])
        configured_games = int(experiment["games"])
        configured_seed = int(experiment["selection_seed"])
        configured_state_pool = dict(experiment["state_pool"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"experiment config is incomplete: {error}") from error
    policy = args.policy or configured_policy
    states_per_game = (
        configured_states_per_game
        if args.states_per_game is None
        else args.states_per_game
    )
    if policy != configured_policy or states_per_game != configured_states_per_game:
        raise SystemExit("selection CLI overrides disagree with the preregistered config")
    seed = configured_seed if args.seed is None else args.seed
    if seed != configured_seed:
        raise SystemExit("--seed disagrees with experiment.selection_seed")
    if policy not in {"uniform_per_game_nested_v1", "top_score_per_game"}:
        raise SystemExit(f"unsupported configured selection policy: {policy}")
    if states_per_game <= 0:
        raise SystemExit("--states-per-game must be positive")
    state_pool_manifest_path = Path(args.state_pool_manifest)
    state_pool_manifest = json.loads(
        state_pool_manifest_path.read_text(encoding="utf-8")
    )
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if (
        not current_revision
        or state_pool_manifest.get("artifact") != "immutable_state_pool"
        or state_pool_manifest.get("protocol_version") != "omniopd-v1"
        or state_pool_manifest.get("code") != current_code
        or state_pool_manifest.get("code_revision") != current_revision
    ):
        raise SystemExit(
            "state-pool manifest must come from this immutable code revision"
        )
    realized_state_pool = {
        "state_source": state_pool_manifest.get("state_source"),
        "games_G": state_pool_manifest.get("games_G"),
        "seed": state_pool_manifest.get("seed"),
        "max_steps": state_pool_manifest.get("max_steps"),
        "max_context_tokens": state_pool_manifest.get("max_context_tokens"),
        "reserve_tokens": state_pool_manifest.get("reserve_tokens"),
        "behavior_max_tokens": state_pool_manifest.get("behavior_max_tokens"),
        "behavior_sampling": state_pool_manifest.get("behavior_sampling"),
    }
    if realized_state_pool != configured_state_pool:
        raise SystemExit(
            "frozen state-pool manifest disagrees with the experiment config: "
            f"realized={realized_state_pool}, configured={configured_state_pool}"
        )
    if state_pool_manifest.get("outputs", {}).get("state_pool_sha256") != sha256_file(
        args.state_pool
    ):
        raise SystemExit("state-pool file does not match its manifest")
    output = Path(args.output)
    manifest_output = Path(args.manifest_output or str(output) + ".manifest.json")
    if output == manifest_output or output.exists() or manifest_output.exists():
        raise SystemExit("selection outputs must be distinct and must not already exist")

    turns = [rollout_turn_from_dict(row) for row in read_jsonl(args.state_pool)]
    hashes = [turn.state.state_hash for turn in turns]
    if not turns or len(hashes) != len(set(hashes)):
        raise SystemExit("state pool must be non-empty and have unique state hashes")
    by_game = defaultdict(list)
    for turn in turns:
        by_game[turn.state.game_id].append(turn)
    if len(by_game) != configured_games:
        raise SystemExit(
            f"state pool contains {len(by_game)} games; experiment requires {configured_games}"
        )

    score_map = None
    if policy == "top_score_per_game":
        if not args.scores or not args.scores_manifest:
            raise SystemExit("top-score selection requires --scores and --scores-manifest")
        if not Path(args.scores).is_file():
            raise SystemExit("--scores does not exist")
        scores_manifest_path = Path(args.scores_manifest)
        if not scores_manifest_path.is_file():
            raise SystemExit("--scores-manifest does not exist")
        scores_manifest = json.loads(
            scores_manifest_path.read_text(encoding="utf-8")
        )
        expected_score_manifest = {
            "artifact": "state_uncertainty_scores",
            "protocol_version": "omniopd-v1",
            "code": current_code,
            "code_revision": current_revision,
            "state_pool_sha256": sha256_file(args.state_pool),
            "state_pool_manifest_sha256": sha256_file(state_pool_manifest_path),
            "scores_sha256": sha256_file(args.scores),
            "target_tokenization": (
                "canonical_final_assistant_content_excluding_terminator"
            ),
            "enable_thinking": False,
        }
        score_mismatches = {
            key: (scores_manifest.get(key), expected)
            for key, expected in expected_score_manifest.items()
            if scores_manifest.get(key) != expected
        }
        if score_mismatches:
            raise SystemExit(
                f"uncertainty scores disagree with their manifest: {score_mismatches}"
            )
        score_rows = list(read_jsonl(args.scores))
        score_map = {str(row["state_hash"]): float(row["score"]) for row in score_rows}
        if len(score_map) != len(score_rows) or set(score_map) != set(hashes):
            raise SystemExit("scores must map one-to-one onto the complete frozen state pool")
        if not all(math.isfinite(score) for score in score_map.values()):
            raise SystemExit("all selection scores must be finite")
    elif args.scores or args.scores_manifest:
        raise SystemExit(
            "--scores/--scores-manifest are only valid for top-score selection"
        )

    selected = []
    for game in sorted(by_game):
        game_turns = sorted(by_game[game], key=lambda turn: turn.state.turn_index)
        if len(game_turns) < states_per_game:
            raise SystemExit(
                f"game {game!r} has {len(game_turns)} states, fewer than requested "
                f"{states_per_game}"
            )
        if policy == "uniform_per_game_nested_v1":
            chosen = select_uniform_nested(
                game_turns,
                states_per_game,
                seed=seed,
                game_id=game,
            )
        else:
            chosen = select_top_score(
                game_turns,
                [score_map[turn.state.state_hash] for turn in game_turns],
                states_per_game,
                policy=policy,
            )
        selected.extend(chosen)

    rows = [
        {
            "protocol_version": "omniopd-v1",
            "state_hash": item.turn.state.state_hash,
            "game_id": item.turn.state.game_id,
            "turn_index": item.turn.state.turn_index,
            "selection_policy": item.policy,
            "inclusion_probability": item.inclusion_probability,
            "score": item.score,
        }
        for item in selected
    ]
    if len(rows) != configured_states:
        raise SystemExit(
            f"selection realized M={len(rows)} but config declares M={configured_states}"
        )
    write_jsonl(output, rows)
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "selection_manifest",
        "code": current_code,
        "code_revision": current_revision,
        "experiment": experiment["experiment"],
        "experiment_config_sha256": sha256_file(experiment_path),
        "state_pool_manifest_sha256": sha256_file(state_pool_manifest_path),
        "policy": policy,
        "seed": seed if policy == "uniform_per_game_nested_v1" else None,
        "games_G": len(by_game),
        "states_per_game": states_per_game,
        "distinct_states_M": len(rows),
        "state_pool_sha256": sha256_file(args.state_pool),
        "score_file_sha256": sha256_file(args.scores) if args.scores else None,
        "score_manifest_sha256": (
            sha256_file(args.scores_manifest) if args.scores_manifest else None
        ),
        "selection_sha256": sha256_file(output),
        "population_inference_supported": policy == "uniform_per_game_nested_v1",
        "uniform_design": (
            "hash_priority_srswor_nested_within_game_v1"
            if policy == "uniform_per_game_nested_v1"
            else None
        ),
        "selected_state_hashes": sorted(row["state_hash"] for row in rows),
        "selected_counts_by_game": {
            game: sum(row["game_id"] == game for row in rows)
            for game in sorted(by_game)
        },
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
