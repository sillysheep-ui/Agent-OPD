#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.io import read_jsonl, rollout_turn_from_dict
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count the finite state population in every frozen game"
    )
    parser.add_argument("--state-pool", required=True)
    parser.add_argument("--state-pool-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output")
    args = parser.parse_args()

    state_pool = Path(args.state_pool)
    state_pool_manifest = Path(args.state_pool_manifest)
    output = Path(args.output)
    manifest_output = Path(args.manifest_output or str(output) + ".manifest.json")
    if not state_pool.is_file() or not state_pool_manifest.is_file():
        raise SystemExit("state pool and state-pool manifest must exist")
    if output == manifest_output or output.exists() or manifest_output.exists():
        raise SystemExit("refusing to overwrite or alias population-count outputs")
    source_manifest = json.loads(state_pool_manifest.read_text(encoding="utf-8"))
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if (
        source_manifest.get("artifact") != "immutable_state_pool"
        or source_manifest.get("protocol_version") != "omniopd-v1"
        or source_manifest.get("outputs", {}).get("state_pool_sha256")
        != sha256_file(state_pool)
        or source_manifest.get("code") != current_code
        or source_manifest.get("code_revision") != current_revision
        or not current_revision
    ):
        raise SystemExit(
            "state pool must match a canonical manifest from this code revision"
        )
    turns = [rollout_turn_from_dict(row) for row in read_jsonl(state_pool)]
    hashes = [turn.state.state_hash for turn in turns]
    if not turns or len(hashes) != len(set(hashes)):
        raise SystemExit("state pool must contain unique non-empty states")
    counts = dict(sorted(Counter(turn.state.game_id for turn in turns).items()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(counts, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "finite_state_population_counts",
        "code": current_code,
        "code_revision": current_revision,
        "state_pool_sha256": sha256_file(state_pool),
        "state_pool_manifest_sha256": sha256_file(state_pool_manifest),
        "states": len(turns),
        "games": len(counts),
        "counts_sha256": sha256_file(output),
    }
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
