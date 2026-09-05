#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.analysis.uncertainty import admissible_action_log_scores
from omniopd.io import read_jsonl, rollout_turn_from_dict, write_jsonl
from omniopd.provenance import (
    fingerprint_code_tree,
    fingerprint_path,
    git_revision,
    sha256_file,
)
from omniopd.selection import admissible_entropy


def _content_identity(value):
    if not isinstance(value, dict):
        return value
    return {
        key: _content_identity(item)
        for key, item in value.items()
        if key not in {"path", "resolved_path", "value"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score state entropy over admissible actions")
    parser.add_argument("--state-pool", required=True)
    parser.add_argument("--state-pool-manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    output = Path(args.output)
    state_pool = Path(args.state_pool)
    state_pool_manifest_path = Path(args.state_pool_manifest)
    manifest_output = Path(args.manifest_output or str(output) + ".manifest.json")
    if output == manifest_output or output.exists() or manifest_output.exists():
        raise SystemExit("score outputs must be distinct and must not already exist")
    if not state_pool.is_file() or not state_pool_manifest_path.is_file():
        raise SystemExit("state pool and immutable-pool manifest must exist")
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    state_pool_manifest = json.loads(
        state_pool_manifest_path.read_text(encoding="utf-8")
    )
    if (
        not current_revision
        or state_pool_manifest.get("artifact") != "immutable_state_pool"
        or state_pool_manifest.get("protocol_version") != "omniopd-v1"
        or state_pool_manifest.get("outputs", {}).get("state_pool_sha256")
        != sha256_file(state_pool)
        or state_pool_manifest.get("code") != current_code
        or state_pool_manifest.get("code_revision") != current_revision
    ):
        raise SystemExit(
            "entropy scoring requires a canonical state pool from this code revision"
        )
    tokenizer_name = args.tokenizer or args.model
    if not Path(args.model).exists() or not Path(tokenizer_name).exists():
        raise SystemExit("--model and --tokenizer must resolve to fingerprintable local paths")
    model_fingerprint = fingerprint_path(args.model)
    tokenizer_fingerprint = fingerprint_path(tokenizer_name)
    behavior_artifacts = state_pool_manifest.get("behavior_artifacts")
    if (
        state_pool_manifest.get("state_source") != "student"
        or not isinstance(behavior_artifacts, list)
        or len(behavior_artifacts) != 1
        or _content_identity(behavior_artifacts[0])
        != _content_identity(model_fingerprint)
        or _content_identity(state_pool_manifest.get("tokenizer"))
        != _content_identity(tokenizer_fingerprint)
    ):
        raise SystemExit(
            "uncertainty must be scored by the exact Student model and tokenizer that "
            "generated the state pool; use one merged fingerprintable behavior artifact"
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise SystemExit("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
    ).to(device)
    model.eval()

    turns = [rollout_turn_from_dict(row) for row in read_jsonl(state_pool)]
    hashes = [turn.state.state_hash for turn in turns]
    if not turns or len(hashes) != len(set(hashes)):
        raise SystemExit("state pool must be non-empty and contain unique hashes")
    rows = []
    for turn in turns:
        scores = admissible_action_log_scores(
            model,
            tokenizer,
            turn.state.messages,
            turn.state.admissible_actions,
            device=device,
        )
        rows.append(
            {
                "protocol_version": "omniopd-v1",
                "state_hash": turn.state.state_hash,
                "game_id": turn.state.game_id,
                "turn_index": turn.state.turn_index,
                "score": admissible_entropy(scores),
                "action_log_scores": dict(zip(turn.state.admissible_actions, scores)),
                "score_definition": "entropy_of_softmax_sequence_action_logprob",
            }
        )
    write_jsonl(output, rows)
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "state_uncertainty_scores",
        "code": current_code,
        "code_revision": current_revision,
        "model": model_fingerprint,
        "tokenizer": tokenizer_fingerprint,
        "device": device,
        "states": len(rows),
        "state_pool_sha256": sha256_file(state_pool),
        "state_pool_manifest_sha256": sha256_file(state_pool_manifest_path),
        "scores_sha256": sha256_file(output),
        "target_tokenization": "canonical_final_assistant_content_excluding_terminator",
        "enable_thinking": False,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
