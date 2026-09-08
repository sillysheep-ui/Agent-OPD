from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from collections.abc import Mapping

from omniopd.evaluation import validate_paired_aggregates
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file
from omniopd.statistics import paired_hierarchical_bootstrap


def _load(path: str) -> dict[int, dict[str, float]]:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("bootstrap aggregate must be a seed-to-game JSON object")
    converted = {}
    for seed, games in raw.items():
        if not isinstance(games, Mapping):
            raise ValueError("every aggregate seed must map to per-game outcomes")
        converted_seed = int(seed)
        if converted_seed in converted:
            raise ValueError("aggregate contains duplicate integer-equivalent seed keys")
        converted[converted_seed] = {
            str(game): float(value) for game, value in games.items()
        }
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired game/seed hierarchical bootstrap")
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--treatment-manifest", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--control-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-seeds", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        raise SystemExit("bootstrap RNG seed must be non-negative")

    paths = {
        "treatment": Path(args.treatment),
        "treatment_manifest": Path(args.treatment_manifest),
        "control": Path(args.control),
        "control_manifest": Path(args.control_manifest),
    }
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite bootstrap result: {output}")
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"bootstrap inputs are missing: {missing}")
    treatment = _load(args.treatment)
    control = _load(args.control)
    if not args.fixed_seeds and len(treatment) < 2:
        raise SystemExit(
            "training-seed uncertainty requires at least two independent seeds; "
            "rerun with --fixed-seeds only if the intended estimand is conditional "
            "on the supplied checkpoint"
        )
    treatment_manifest = json.loads(
        paths["treatment_manifest"].read_text(encoding="utf-8")
    )
    control_manifest = json.loads(
        paths["control_manifest"].read_text(encoding="utf-8")
    )
    for arm, manifest in [
        ("treatment", treatment_manifest),
        ("control", control_manifest),
    ]:
        actual = sha256_file(paths[arm])
        if manifest.get("output_sha256") != actual:
            raise SystemExit(f"{arm} aggregate does not match its manifest")
    pairing = validate_paired_aggregates(
        treatment, control, treatment_manifest, control_manifest
    )
    if len(pairing["games"]) < 2:
        raise SystemExit(
            "game-cluster uncertainty requires at least two paired evaluation games"
        )
    repository_root = Path(__file__).resolve().parents[3]
    current_code = fingerprint_code_tree(repository_root)
    current_revision = git_revision(repository_root)
    if (
        pairing["evaluation_contract"].get("code") != current_code
        or pairing["evaluation_contract"].get("code_revision") != current_revision
    ):
        raise SystemExit(
            "evaluation, aggregation, and bootstrap must use one code revision"
        )
    result = paired_hierarchical_bootstrap(
        treatment,
        control,
        replicates=args.replicates,
        confidence=args.confidence,
        rng_seed=args.seed,
        resample_seeds=not args.fixed_seeds,
    )
    payload = {
        "protocol_version": "omniopd-v1",
        "artifact": "paired_hierarchical_bootstrap_result",
        "interpretation": (
            "training_seed_and_paired_game_uncertainty"
            if not args.fixed_seeds
            else "paired_game_uncertainty_conditional_on_fixed_checkpoints"
        ),
        "code": current_code,
        "code_revision": current_revision,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "pairing": pairing,
        "bootstrap_rng_seed": args.seed,
        "result": asdict(result),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
