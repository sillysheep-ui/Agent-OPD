#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.features import state_diagnostic_features
from omniopd.io import correction_from_dict, read_jsonl, rollout_turn_from_dict
from omniopd.provenance import (
    fingerprint_code_tree,
    git_revision,
    sha256_file,
)
from omniopd.representativeness import paired_game_balanced_bootstrap_js
from omniopd.validation import audit_correction_records


def _parse_group(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("group must be NAME=CORRECTIONS.jsonl")
    name, path = value.split("=", 1)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
        raise argparse.ArgumentTypeError("group name is not a safe identifier")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M2 game-balanced marginal representativeness diagnostics"
    )
    parser.add_argument("--state-pool", required=True)
    parser.add_argument("--state-pool-manifest", required=True)
    parser.add_argument("--group", action="append", type=_parse_group, required=True)
    parser.add_argument(
        "--group-manifest", action="append", type=_parse_group, required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    state_pool = Path(args.state_pool)
    state_pool_manifest_path = Path(args.state_pool_manifest)
    output = Path(args.output)
    groups = dict(args.group)
    group_manifests = dict(args.group_manifest)
    if len(groups) != len(args.group) or len(group_manifests) != len(
        args.group_manifest
    ):
        raise SystemExit("M2 group and group-manifest names must be unique")
    if set(groups) != set(group_manifests):
        raise SystemExit("M2 groups and group manifests must have identical names")
    all_inputs = [
        state_pool,
        state_pool_manifest_path,
        *groups.values(),
        *group_manifests.values(),
    ]
    if any(not path.is_file() for path in all_inputs):
        raise SystemExit("state pool, corrections, and all manifests must exist")
    if output.exists():
        raise SystemExit(f"refusing to overwrite M2 output: {output}")
    if args.max_steps <= 0 or args.replicates <= 0:
        raise SystemExit("--max-steps and --replicates must be positive")
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    state_pool_sha256 = sha256_file(state_pool)
    state_pool_manifest = json.loads(
        state_pool_manifest_path.read_text(encoding="utf-8")
    )
    if (
        not current_revision
        or state_pool_manifest.get("artifact") != "immutable_state_pool"
        or state_pool_manifest.get("protocol_version") != "omniopd-v1"
        or state_pool_manifest.get("outputs", {}).get("state_pool_sha256")
        != state_pool_sha256
        or state_pool_manifest.get("code") != current_code
        or state_pool_manifest.get("code_revision") != current_revision
    ):
        raise SystemExit(
            "M2 state pool must match a canonical manifest from this code revision"
        )
    if int(state_pool_manifest.get("max_steps", -1)) != args.max_steps:
        raise SystemExit("--max-steps must equal the frozen state-pool rollout horizon")

    turns = [rollout_turn_from_dict(row) for row in read_jsonl(state_pool)]
    hashes = [turn.state.state_hash for turn in turns]
    if not turns or len(hashes) != len(set(hashes)):
        raise SystemExit("state pool must be non-empty and contain unique state hashes")
    by_game: dict[str, list] = defaultdict(list)
    for turn in turns:
        by_game[turn.state.game_id].append(turn)
    features_by_hash = {}
    reference: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    feature_names = None
    for game, game_turns in by_game.items():
        previous = None
        for turn in sorted(game_turns, key=lambda item: item.state.turn_index):
            features = state_diagnostic_features(
                turn,
                previous_observation=previous,
                max_steps=args.max_steps,
            )
            previous = turn.state.observation
            features_by_hash[turn.state.state_hash] = features
            feature_names = sorted(features)
            for feature, value in features.items():
                reference[feature][game].append(value)

    group_reports = {}
    group_input_metadata = {}
    for group_name, path in sorted(groups.items()):
        group_manifest_path = group_manifests[group_name]
        group_manifest = json.loads(
            group_manifest_path.read_text(encoding="utf-8")
        )
        if (
            group_manifest.get("artifact") != "teacher_corrections"
            or group_manifest.get("protocol_version") != "omniopd-v1"
            or group_manifest.get("corrections_sha256") != sha256_file(path)
            or group_manifest.get("state_pool_sha256") != state_pool_sha256
            or group_manifest.get("code") != current_code
            or group_manifest.get("code_revision") != current_revision
        ):
            raise SystemExit(
                f"M2 group {group_name} is not bound to the canonical state pool"
            )
        records = [correction_from_dict(row) for row in read_jsonl(path)]
        issues = audit_correction_records(records)
        if issues:
            raise SystemExit(
                f"M2 group {group_name} correction audit failed: {issues[:5]}"
            )
        state_hashes = [record.state.state_hash for record in records]
        if len(state_hashes) != len(set(state_hashes)):
            raise SystemExit(f"M2 group {group_name} contains duplicate states")
        unknown = sorted(set(state_hashes) - set(features_by_hash))
        if unknown:
            raise SystemExit(
                f"M2 group {group_name} contains states outside the frozen pool: {unknown[:5]}"
            )
        selected: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        retained = [record for record in records if record.valid_teacher_samples]
        for record in retained:
            features = features_by_hash[record.state.state_hash]
            for feature, value in features.items():
                selected[feature][record.state.game_id].append(value)
        retained_games = sorted({record.state.game_id for record in retained})
        dropped_games = sorted(set(by_game) - set(retained_games))
        diagnostics = {}
        if retained_games:
            for feature in feature_names or []:
                reference_common = {
                    game: reference[feature][game] for game in retained_games
                }
                selected_common = {
                    game: selected[feature][game] for game in retained_games
                }
                interval = paired_game_balanced_bootstrap_js(
                    reference_common,
                    selected_common,
                    replicates=args.replicates,
                    rng_seed=args.seed,
                )
                diagnostics[feature] = interval.__dict__
        group_reports[group_name] = {
            "selected_states": len(records),
            "teacher_valid_states": len(retained),
            "retained_games": retained_games,
            "dropped_games": dropped_games,
            "estimand": "game_balanced_conditional_on_retained_games",
            "marginal_js": diagnostics,
        }
        group_input_metadata[group_name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "manifest_path": str(group_manifest_path),
            "manifest_sha256": sha256_file(group_manifest_path),
        }

    report = {
        "protocol_version": "omniopd-v1",
        "artifact": "m2_representativeness_diagnostic",
        "interpretation": "marginal_distribution_diagnostic_not_causal_utility",
        "code": current_code,
        "code_revision": current_revision,
        "state_pool_sha256": state_pool_sha256,
        "state_pool_manifest_sha256": sha256_file(state_pool_manifest_path),
        "state_pool_states": len(turns),
        "state_pool_games": len(by_game),
        "group_inputs": group_input_metadata,
        "bootstrap_replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "feature_contract": {
            "max_steps": args.max_steps,
            "turn_depth_bins": 10,
            "admissible_count_bins": 10,
            "maximum_admissible_count": 50,
        },
        "groups": group_reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
