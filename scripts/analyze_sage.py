#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.analysis.sage import (
    design_weighted_game_balanced_indicator,
    deterministic_or_judged_label,
    disagreement,
    game_cluster_bootstrap_design_ratio,
    game_balanced_indicator,
    state_level_disagreement,
)
from omniopd.io import read_jsonl, write_jsonl
from omniopd.provenance import (
    fingerprint_code_tree,
    git_revision,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize completed SAGE labels with design-based game balance"
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--labels-manifest", required=True)
    parser.add_argument(
        "--population-counts",
        required=True,
        help="JSON object mapping every target game_id to its state count",
    )
    parser.add_argument("--population-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels_path = Path(args.labels)
    labels_manifest_path = Path(args.labels_manifest)
    population_path = Path(args.population_counts)
    population_manifest_path = Path(args.population_manifest)
    output = Path(args.output_dir)
    if (
        not labels_path.is_file()
        or not labels_manifest_path.is_file()
        or not population_path.is_file()
        or not population_manifest_path.is_file()
    ):
        raise SystemExit("SAGE labels, label manifest, and population counts must exist")
    if output.exists():
        raise SystemExit(f"refusing to overwrite SAGE output: {output}")
    if args.replicates <= 0:
        raise SystemExit("--replicates must be positive")
    population = {
        str(game): int(count)
        for game, count in json.loads(
            population_path.read_text(encoding="utf-8")
        ).items()
    }
    if not population or any(count <= 0 for count in population.values()):
        raise SystemExit("population counts must be a non-empty positive mapping")
    population_manifest = json.loads(
        population_manifest_path.read_text(encoding="utf-8")
    )
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if (
        population_manifest.get("artifact") != "finite_state_population_counts"
        or population_manifest.get("counts_sha256") != sha256_file(population_path)
        or population_manifest.get("code") != current_code
        or population_manifest.get("code_revision") != current_revision
    ):
        raise SystemExit("population counts do not match their frozen-pool manifest")
    labels_manifest = json.loads(labels_manifest_path.read_text(encoding="utf-8"))
    if (
        labels_manifest.get("artifact") != "sage_blind_intervention_labels"
        or labels_manifest.get("labels_sha256") != sha256_file(labels_path)
        or labels_manifest.get("teacher_information_sent_to_judge") is not False
        or labels_manifest.get("state_pool_sha256")
        != population_manifest.get("state_pool_sha256")
        or labels_manifest.get("code") != current_code
        or labels_manifest.get("code_revision") != current_revision
        or not isinstance(labels_manifest.get("experiment"), str)
        or not labels_manifest["experiment"].strip()
    ):
        raise SystemExit("SAGE labels do not match a canonical blind-judge manifest")

    source_rows = list(read_jsonl(labels_path))
    required = {
        "state_hash",
        "game_id",
        "student_action",
        "student_valid",
        "teacher_samples",
        "inclusion_probability",
    }
    if not source_rows or any(required - set(row) for row in source_rows):
        raise SystemExit(f"every SAGE label row requires {sorted(required)}")
    state_hashes = [str(row["state_hash"]) for row in source_rows]
    if len(state_hashes) != len(set(state_hashes)):
        raise SystemExit("SAGE label rows must contain unique states")

    processed = []
    for row in source_rows:
        teacher_samples = row["teacher_samples"]
        if not isinstance(teacher_samples, list) or not teacher_samples:
            raise SystemExit(
                "every SAGE state must retain its non-empty Teacher-attempt list"
            )
        student_valid = bool(row["student_valid"])
        try:
            label = deterministic_or_judged_label(
                student_valid=student_valid,
                judge_label=row.get("judge_label"),
            )
        except ValueError as error:
            raise SystemExit(
                f"state {row['state_hash']} contains a non-canonical SAGE label"
            ) from error
        disagreement_values = [
            value
            for value in (
                disagreement(str(row["student_action"]), sample)
                for sample in teacher_samples
            )
            if value is not None
        ]
        processed.append(
            {
                "state_hash": str(row["state_hash"]),
                "game_id": str(row["game_id"]),
                "inclusion_probability": row["inclusion_probability"],
                "student_valid": student_valid,
                "sage_label": label,
                "skip": int(label == "Skip") if label is not None else None,
                "weak": int(label == "Weak") if label is not None else None,
                "strong": int(label == "Strong") if label is not None else None,
                "teacher_valid": bool(disagreement_values),
                "disagreement": state_level_disagreement(
                    str(row["student_action"]), teacher_samples
                ),
                "valid_teacher_samples": len(disagreement_values),
                "total_teacher_samples": len(teacher_samples),
            }
        )

    summaries = {}
    for indicator in ["skip", "weak", "strong"]:
        try:
            observed = game_balanced_indicator(processed, indicator_key=indicator)
        except ValueError:
            observed = None
        population_estimate = None
        population_nonidentification = None
        try:
            population_estimate = design_weighted_game_balanced_indicator(
                processed,
                indicator_key=indicator,
                population_states_by_game=population,
            )
        except ValueError as error:
            population_nonidentification = str(error)
        summaries[indicator] = {
            "observed_sample_game_balanced": observed.__dict__ if observed else None,
            "design_weighted_population": (
                population_estimate.__dict__ if population_estimate else None
            ),
            "population_nonidentification": population_nonidentification,
        }
    observed_disagreement = [
        row for row in processed if row["disagreement"] is not None
    ]
    disagreement_summary = (
        game_balanced_indicator(observed_disagreement, indicator_key="disagreement").__dict__
        if observed_disagreement
        else None
    )

    def ratio_summary(numerator, denominator, *, estimand: str):
        rows = []
        for row in processed:
            numerator_value, denominator_value = numerator(row), denominator(row)
            rows.append(
                {
                    **row,
                    "_numerator": numerator_value,
                    "_denominator": denominator_value,
                }
            )
        try:
            interval = game_cluster_bootstrap_design_ratio(
                rows,
                numerator_key="_numerator",
                denominator_key="_denominator",
                population_states_by_game=population,
                replicates=args.replicates,
                rng_seed=args.seed,
            )
            return {
                "estimand": estimand,
                "design_weighted_population": interval.__dict__,
                "population_nonidentification": None,
            }
        except ValueError as error:
            return {
                "estimand": estimand,
                "design_weighted_population": None,
                "population_nonidentification": str(error),
            }

    paper_conditionals = {"P(I|D)": {}, "P(D|I)": {}, "P(V_T|I)": {}}
    for d_value in [0, 1]:
        for label_value in ["Skip", "Weak", "Strong"]:
            key = f"I={label_value}|D={d_value}"

            def denominator_i_d(row, d=d_value):
                return int(
                    row["student_valid"]
                    and row["teacher_valid"]
                    and row["disagreement"] == d
                )

            def numerator_i_d(row, d=d_value, label=label_value):
                denom = denominator_i_d(row, d)
                if denom and row["sage_label"] is None:
                    return None
                return int(denom and row["sage_label"] == label)

            paper_conditionals["P(I|D)"][key] = ratio_summary(
                numerator_i_d,
                denominator_i_d,
                estimand="P(I|D,student_valid=1,V_T=1)",
            )
    for label_value in ["Skip", "Weak", "Strong"]:
        for d_value in [0, 1]:
            key = f"D={d_value}|I={label_value}"

            def denominator_d_i(row, label=label_value):
                if row["student_valid"] and row["sage_label"] is None:
                    return None
                return int(
                    row["student_valid"]
                    and row["teacher_valid"]
                    and row["sage_label"] == label
                )

            def numerator_d_i(row, d=d_value, label=label_value):
                denom = denominator_d_i(row, label)
                return None if denom is None else int(denom and row["disagreement"] == d)

            paper_conditionals["P(D|I)"][key] = ratio_summary(
                numerator_d_i,
                denominator_d_i,
                estimand="P(D|I,student_valid=1,V_T=1)",
            )
        def denominator_v_i(row, label=label_value):
            if row["student_valid"] and row["sage_label"] is None:
                return None
            return int(row["sage_label"] == label)

        def numerator_v_i(row, label=label_value):
            denom = denominator_v_i(row, label)
            return None if denom is None else int(denom and row["teacher_valid"])

        paper_conditionals["P(V_T|I)"][f"V_T=1|I={label_value}"] = ratio_summary(
            numerator_v_i,
            denominator_v_i,
            estimand="P(V_T=1|I)",
        )

    def u_denominator(row):
        return int(row["student_valid"])

    def u_numerator(row):
        if row["student_valid"] and row["sage_label"] is None:
            return None
        return int(
            row["student_valid"]
            and row["teacher_valid"]
            and row["disagreement"] == 1
            and row["sage_label"] == "Skip"
        )

    group_identity = labels_manifest.get("experiment")
    u_g = ratio_summary(
        u_numerator,
        u_denominator,
        estimand="P(D=1,I=Skip|student_valid=1,g)",
    )
    u_g["group"] = group_identity

    output.mkdir(parents=True)
    processed_path = output / "processed_labels.jsonl"
    write_jsonl(processed_path, processed)
    report = {
        "protocol_version": "omniopd-v1",
        "artifact": "sage_design_weighted_summary",
        "code": current_code,
        "code_revision": current_revision,
        "labels_sha256": sha256_file(labels_path),
        "labels_manifest_sha256": sha256_file(labels_manifest_path),
        "population_counts_sha256": sha256_file(population_path),
        "population_manifest_sha256": sha256_file(population_manifest_path),
        "processed_labels_sha256": sha256_file(processed_path),
        "states": len(processed),
        "technical_states_retained": sum(not row["student_valid"] for row in processed),
        "missing_executable_judge_labels": sum(
            row["student_valid"] and row["sage_label"] is None for row in processed
        ),
        "teacher_invalid_disagreement_missing": sum(
            row["disagreement"] is None for row in processed
        ),
        "sage_proportions": summaries,
        "disagreement_observed_only": disagreement_summary,
        "paper_conditionals": paper_conditionals,
        "U_g": u_g,
        "bootstrap_replicates": args.replicates,
        "bootstrap_seed": args.seed,
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
