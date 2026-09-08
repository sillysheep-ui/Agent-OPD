#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.analysis.sage import (
    design_weighted_game_balanced_indicator,
    deterministic_or_judged_label,
    disagreement,
    game_cluster_bootstrap_design_ratio,
    game_balanced_indicator,
    observed_sample_game_balanced_ratio,
    paired_game_cluster_bootstrap_indicator_difference,
    state_level_disagreement,
)
from omniopd.io import read_jsonl, write_jsonl
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file


Indicator = Callable[[dict[str, Any]], int | None]


def _ratio_summary(
    rows: list[dict[str, Any]],
    numerator: Indicator,
    denominator: Indicator,
    *,
    estimand: str,
    population: dict[str, int],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    calculation_rows = [
        {
            **row,
            "_numerator": numerator(row),
            "_denominator": denominator(row),
        }
        for row in rows
    ]
    observed = None
    observed_nonidentification = None
    try:
        observed = observed_sample_game_balanced_ratio(
            calculation_rows,
            numerator_key="_numerator",
            denominator_key="_denominator",
        )
    except ValueError as error:
        observed_nonidentification = str(error)
    population_interval = None
    population_nonidentification = None
    try:
        population_interval = game_cluster_bootstrap_design_ratio(
            calculation_rows,
            numerator_key="_numerator",
            denominator_key="_denominator",
            population_states_by_game=population,
            replicates=replicates,
            rng_seed=seed,
        )
    except ValueError as error:
        population_nonidentification = str(error)
    return {
        "estimand": estimand,
        "observed_selected_sample_game_balanced": (
            observed.__dict__ if observed else None
        ),
        "observed_sample_nonidentification": observed_nonidentification,
        "design_weighted_population": (
            population_interval.__dict__ if population_interval else None
        ),
        "population_nonidentification": population_nonidentification,
        "population_uncertainty_scope": (
            "game-cluster bootstrap conditional on the realized within-game state "
            "selection; within-game SRSWOR design variance is not propagated"
            if population_interval
            else None
        ),
    }


def _summarize_group(
    rows: list[dict[str, Any]],
    *,
    group: str,
    population: dict[str, int],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    proportions: dict[str, Any] = {}
    for indicator in ["skip", "weak", "strong"]:
        try:
            observed = game_balanced_indicator(rows, indicator_key=indicator)
        except ValueError:
            observed = None
        population_estimate = None
        population_nonidentification = None
        try:
            population_estimate = design_weighted_game_balanced_indicator(
                rows,
                indicator_key=indicator,
                population_states_by_game=population,
            )
        except ValueError as error:
            population_nonidentification = str(error)
        proportions[indicator] = {
            "observed_selected_sample_game_balanced": (
                observed.__dict__ if observed else None
            ),
            "design_weighted_population": (
                population_estimate.__dict__ if population_estimate else None
            ),
            "population_nonidentification": population_nonidentification,
        }

    disagreement_rows = [row for row in rows if row["disagreement"] is not None]
    disagreement_summary = (
        game_balanced_indicator(
            disagreement_rows, indicator_key="disagreement"
        ).__dict__
        if disagreement_rows
        else None
    )

    conditionals: dict[str, dict[str, Any]] = {
        "P(I|D)": {},
        "P(D|I)": {},
        "P(V_T|I)": {},
    }
    for d_value in [0, 1]:
        for label_value in ["Skip", "Weak", "Strong"]:
            key = f"I={label_value}|D={d_value}"

            def denominator_i_d(row: dict[str, Any], d: int = d_value) -> int:
                return int(
                    row["student_valid"]
                    and row["teacher_valid"]
                    and row["disagreement"] == d
                )

            def numerator_i_d(
                row: dict[str, Any],
                d: int = d_value,
                label: str = label_value,
            ) -> int | None:
                denom = denominator_i_d(row, d)
                if denom and row["sage_label"] is None:
                    return None
                return int(denom and row["sage_label"] == label)

            conditionals["P(I|D)"][key] = _ratio_summary(
                rows,
                numerator_i_d,
                denominator_i_d,
                estimand=f"P(I|D,student_valid=1,V_T=1,g={group})",
                population=population,
                replicates=replicates,
                seed=seed,
            )

    for label_value in ["Skip", "Weak", "Strong"]:
        for d_value in [0, 1]:
            key = f"D={d_value}|I={label_value}"

            def denominator_d_i(
                row: dict[str, Any], label: str = label_value
            ) -> int | None:
                # Missing I matters only inside the declared V_T=1 conditioning set.
                if (
                    row["student_valid"]
                    and row["teacher_valid"]
                    and row["sage_label"] is None
                ):
                    return None
                return int(
                    row["student_valid"]
                    and row["teacher_valid"]
                    and row["sage_label"] == label
                )

            def numerator_d_i(
                row: dict[str, Any],
                d: int = d_value,
                label: str = label_value,
            ) -> int | None:
                denom = denominator_d_i(row, label)
                return None if denom is None else int(
                    denom and row["disagreement"] == d
                )

            conditionals["P(D|I)"][key] = _ratio_summary(
                rows,
                numerator_d_i,
                denominator_d_i,
                estimand=f"P(D|I,student_valid=1,V_T=1,g={group})",
                population=population,
                replicates=replicates,
                seed=seed,
            )

        def denominator_v_i(
            row: dict[str, Any], label: str = label_value
        ) -> int | None:
            if row["student_valid"] and row["sage_label"] is None:
                return None
            return int(row["sage_label"] == label)

        def numerator_v_i(
            row: dict[str, Any], label: str = label_value
        ) -> int | None:
            denom = denominator_v_i(row, label)
            return None if denom is None else int(denom and row["teacher_valid"])

        conditionals["P(V_T|I)"][f"V_T=1|I={label_value}"] = _ratio_summary(
            rows,
            numerator_v_i,
            denominator_v_i,
            estimand=f"P(V_T=1|I,g={group})",
            population=population,
            replicates=replicates,
            seed=seed,
        )

    def u_all_selected_denominator(_row: dict[str, Any]) -> int:
        return 1

    def u_executable_denominator(row: dict[str, Any]) -> int:
        return int(row["student_valid"])

    def u_executable_numerator(row: dict[str, Any]) -> int | None:
        if (
            row["student_valid"]
            and row["teacher_valid"]
            and row["disagreement"] == 1
        ):
            if row["sage_label"] is None:
                return None
        return int(
            row["student_valid"]
            and row["teacher_valid"]
            and row["disagreement"] == 1
            and row["sage_label"] == "Skip"
        )

    def u_all_selected_numerator(row: dict[str, Any]) -> int | None:
        if row["teacher_valid"] and row["disagreement"] == 1:
            if row["sage_label"] is None:
                return None
        return int(
            row["teacher_valid"]
            and row["disagreement"] == 1
            and row["sage_label"] == "Skip"
        )

    def u_valid_denominator(row: dict[str, Any]) -> int:
        return int(row["teacher_valid"])

    return {
        "group": group,
        "memberships": len(rows),
        "unique_states": len({row["state_hash"] for row in rows}),
        "sage_proportions": proportions,
        "disagreement_observed_selected_sample": disagreement_summary,
        "paper_conditionals": conditionals,
        "primary_estimand": "U_g_executable",
        "U_g_executable": _ratio_summary(
            rows,
            u_executable_numerator,
            u_executable_denominator,
            estimand=(
                f"P(V_T=1,D=1,I=Skip|student_valid=1,g={group})"
            ),
            population=population,
            replicates=replicates,
            seed=seed,
        ),
        "U_g_all_selected_secondary": _ratio_summary(
            rows,
            u_all_selected_numerator,
            u_all_selected_denominator,
            estimand=f"P(V_T=1,D=1,I=Skip|g={group})",
            population=population,
            replicates=replicates,
            seed=seed,
        ),
        "U_g_teacher_valid_conditional_secondary": _ratio_summary(
            rows,
            u_all_selected_numerator,
            u_valid_denominator,
            estimand=f"P(D=1,I=Skip|V_T=1,g={group})",
            population=population,
            replicates=replicates,
            seed=seed,
        ),
    }


def _strong_a3_minus_a1_contrast(
    by_group: dict[str, list[dict[str, Any]]],
    *,
    population: dict[str, int],
    replicates: int,
    seed: int,
    reference_group: str = "A1",
    comparison_group: str = "A3",
) -> dict[str, Any]:
    """Return a preregistered comparison-reference Strong contrast."""

    contrast = f"{comparison_group}_minus_{reference_group}"
    if (
        not reference_group
        or not comparison_group
        or reference_group == comparison_group
        or not {reference_group, comparison_group} <= set(by_group)
    ):
        return {
            "contrast": contrast,
            "common_game_support": [],
            "observed_selected_sample": None,
            "observed_nonidentification": (
                "two distinct declared contrast memberships are required"
            ),
            "design_weighted_population": None,
            "population_nonidentification": (
                "two distinct declared contrast memberships are required"
            ),
        }
    reference_rows = by_group[reference_group]
    comparison_rows = by_group[comparison_group]
    common_games = sorted(
        {str(row["game_id"]) for row in reference_rows}
        & {str(row["game_id"]) for row in comparison_rows}
    )
    observed = None
    observed_nonidentification = None
    try:
        observed = paired_game_cluster_bootstrap_indicator_difference(
            reference_rows,
            comparison_rows,
            indicator_key="strong",
            replicates=replicates,
            rng_seed=seed,
        )
    except ValueError as error:
        observed_nonidentification = str(error)
    population_estimate = None
    population_nonidentification = None
    try:
        population_estimate = paired_game_cluster_bootstrap_indicator_difference(
            reference_rows,
            comparison_rows,
            indicator_key="strong",
            population_states_by_game=population,
            replicates=replicates,
            rng_seed=seed,
        )
    except ValueError as error:
        population_nonidentification = str(error)
    return {
        "contrast": contrast,
        "reference_group": reference_group,
        "comparison_group": comparison_group,
        "indicator": "I=Strong",
        "common_game_support": common_games,
        "observed_selected_sample": observed.__dict__ if observed else None,
        "observed_nonidentification": observed_nonidentification,
        "design_weighted_population": (
            population_estimate.__dict__ if population_estimate else None
        ),
        "population_nonidentification": population_nonidentification,
        "observed_uncertainty_scope": (
            "paired game-cluster bootstrap on the realized selected memberships "
            "of common games"
            if observed
            else None
        ),
        "population_uncertainty_scope": (
            "paired game-cluster bootstrap on common games, conditional on the "
            "realized within-game state selection; within-game design variance is "
            "not propagated"
            if population_estimate
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize union-deduplicated SAGE labels by experiment membership"
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
    parser.add_argument("--contrast-reference", default="A1")
    parser.add_argument("--contrast-comparison", default="A3")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    labels_manifest_path = Path(args.labels_manifest)
    population_path = Path(args.population_counts)
    population_manifest_path = Path(args.population_manifest)
    output = Path(args.output_dir)
    required_paths = [
        labels_path,
        labels_manifest_path,
        population_path,
        population_manifest_path,
    ]
    if any(not path.is_file() for path in required_paths):
        raise SystemExit("SAGE labels, manifest, and population counts must exist")
    if output.exists():
        raise SystemExit(f"refusing to overwrite SAGE output: {output}")
    if (
        args.replicates <= 0
        or not args.contrast_reference.strip()
        or not args.contrast_comparison.strip()
        or args.contrast_reference == args.contrast_comparison
    ):
        raise SystemExit(
            "replicates must be positive and contrast groups non-empty/distinct"
        )

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
        not current_revision
        or population_manifest.get("artifact") != "finite_state_population_counts"
        or population_manifest.get("counts_sha256") != sha256_file(population_path)
        or population_manifest.get("code") != current_code
        or population_manifest.get("code_revision") != current_revision
    ):
        raise SystemExit("population counts do not match their frozen-pool manifest")

    labels_manifest = json.loads(labels_manifest_path.read_text(encoding="utf-8"))
    experiments = labels_manifest.get("experiments")
    if (
        labels_manifest.get("artifact") != "sage_blind_intervention_labels"
        or labels_manifest.get("labels_sha256") != sha256_file(labels_path)
        or labels_manifest.get("teacher_information_sent_to_judge") is not False
        or labels_manifest.get("state_pool_sha256")
        != population_manifest.get("state_pool_sha256")
        or labels_manifest.get("code") != current_code
        or labels_manifest.get("code_revision") != current_revision
        or labels_manifest.get("sage_input_schema")
        != "union_unique_state_memberships_v1"
        or not isinstance(experiments, list)
        or not experiments
        or len(experiments) != len(set(experiments))
        or int(labels_manifest.get("teacher_samples_per_membership_N", -1)) != 1
        or labels_manifest.get("disagreement_definition")
        != "single_teacher_draw_action_inequality"
    ):
        raise SystemExit("SAGE labels do not match a canonical union blind-judge manifest")

    source_rows = list(read_jsonl(labels_path))
    required = {
        "state_hash",
        "game_id",
        "student_action",
        "student_valid",
        "memberships",
    }
    if not source_rows or any(required - set(row) for row in source_rows):
        raise SystemExit(f"every SAGE label row requires {sorted(required)}")
    state_hashes = [str(row["state_hash"]) for row in source_rows]
    if len(state_hashes) != len(set(state_hashes)):
        raise SystemExit("SAGE blind labels must contain each union state exactly once")

    processed: list[dict[str, Any]] = []
    membership_ids: set[str] = set()
    for row in source_rows:
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
        memberships = row["memberships"]
        if not isinstance(memberships, list) or not memberships:
            raise SystemExit("every union state must retain one or more group memberships")
        for membership in memberships:
            group = str(membership.get("experiment", ""))
            teacher_samples = membership.get("teacher_samples")
            membership_id = f"{group}:{row['state_hash']}"
            if (
                group not in experiments
                or membership_id in membership_ids
                or not isinstance(teacher_samples, list)
                or len(teacher_samples) != 1
            ):
                raise SystemExit("invalid or duplicate SAGE group membership")
            membership_ids.add(membership_id)
            disagreement_values = [
                value
                for value in (
                    disagreement(str(row["student_action"]), sample)
                    for sample in teacher_samples
                )
                if value is not None
            ]
            computed_disagreement = state_level_disagreement(
                str(row["student_action"]), teacher_samples
            )
            if (
                membership.get("teacher_valid") != bool(disagreement_values)
                or membership.get("disagreement") != computed_disagreement
            ):
                raise SystemExit("stored SAGE D/V_T fields disagree with Teacher sample")
            processed.append(
                {
                    "membership_id": membership_id,
                    "experiment": group,
                    "state_hash": str(row["state_hash"]),
                    "game_id": str(row["game_id"]),
                    "inclusion_probability": membership.get(
                        "inclusion_probability"
                    ),
                    "selection_policy": membership.get("selection_policy"),
                    "student_valid": student_valid,
                    "sage_label": label,
                    "skip": int(label == "Skip") if label is not None else None,
                    "weak": int(label == "Weak") if label is not None else None,
                    "strong": int(label == "Strong") if label is not None else None,
                    "teacher_valid": bool(disagreement_values),
                    "disagreement": computed_disagreement,
                    "valid_teacher_samples": len(disagreement_values),
                    "total_teacher_samples": 1,
                }
            )
    if {row["experiment"] for row in processed} != set(experiments):
        raise SystemExit("labels do not realize every declared SAGE experiment group")

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in processed:
        by_group[row["experiment"]].append(row)
    group_reports = {
        group: _summarize_group(
            by_group[group],
            group=group,
            population=population,
            replicates=args.replicates,
            seed=args.seed,
        )
        for group in sorted(by_group)
    }
    strong_contrast = _strong_a3_minus_a1_contrast(
        by_group,
        population=population,
        replicates=args.replicates,
        seed=args.seed,
        reference_group=args.contrast_reference,
        comparison_group=args.contrast_comparison,
    )

    unique_label_rows = []
    for row in source_rows:
        label = deterministic_or_judged_label(
            student_valid=bool(row["student_valid"]),
            judge_label=row.get("judge_label"),
        )
        unique_label_rows.append(
            {
                "game_id": str(row["game_id"]),
                "skip": int(label == "Skip") if label is not None else None,
                "weak": int(label == "Weak") if label is not None else None,
                "strong": int(label == "Strong") if label is not None else None,
            }
        )
    unique_state_label_descriptives = {}
    for indicator in ["skip", "weak", "strong"]:
        try:
            value = game_balanced_indicator(
                unique_label_rows, indicator_key=indicator
            )
        except ValueError:
            value = None
        unique_state_label_descriptives[indicator] = (
            value.__dict__ if value else None
        )

    output.mkdir(parents=True)
    processed_path = output / "processed_memberships.jsonl"
    write_jsonl(processed_path, processed)
    report = {
        "protocol_version": "omniopd-v1",
        "artifact": "sage_union_membership_summary",
        "code": current_code,
        "code_revision": current_revision,
        "labels_sha256": sha256_file(labels_path),
        "labels_manifest_sha256": sha256_file(labels_manifest_path),
        "population_counts_sha256": sha256_file(population_path),
        "population_manifest_sha256": sha256_file(population_manifest_path),
        "processed_memberships_sha256": sha256_file(processed_path),
        "unique_states_judged_once": len(source_rows),
        "group_memberships": len(processed),
        "overlapping_memberships": len(processed) - len(source_rows),
        "technical_unique_states_deterministic_strong": sum(
            not bool(row["student_valid"]) for row in source_rows
        ),
        "missing_executable_unique_state_labels": sum(
            bool(row["student_valid"]) and row.get("judge_label") is None
            for row in source_rows
        ),
        "teacher_invalid_membership_D_missing": sum(
            row["disagreement"] is None for row in processed
        ),
        "unique_state_label_observed_descriptives": unique_state_label_descriptives,
        "groups": group_reports,
        "preregistered_group_contrast": strong_contrast,
        "bootstrap_replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "population_inference_note": (
            "Design-weighted fields require known non-zero inclusion probabilities. "
            "Observed selected-sample fields remain descriptive for deterministic top-k."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
