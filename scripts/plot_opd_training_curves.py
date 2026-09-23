#!/usr/bin/env python3
"""Plot the per-step training curves of a veRL OPD run from its log.

Reads the console log that `scripts/run_verl_opd_train.sh` tees, extracts the
per-step metrics, and draws them with a moving average next to the raw trace so
a flat trend is not mistaken for noise and vice versa.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

METRICS = (
    ("actor/distillation/loss", "distillation loss (k2, per action token)", "#1f77b4"),
    ("actor/entropy", "policy entropy at supervised tokens", "#d62728"),
    ("actor/grad_norm", "gradient norm", "#2ca02c"),
    ("response_length/mean", "mean response length (tokens)", "#7f7f7f"),
)


def parse_steps(log_path: Path) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    pattern = re.compile(r"step:(\d+) - ")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        step = int(match.group(1))
        values = {}
        for key, _, _ in METRICS:
            found = re.search(re.escape(key) + r":(-?[0-9.eE+]+)", line)
            if found is not None:
                values[key] = float(found.group(1))
        if values:
            steps[step] = values
    if not steps:
        raise SystemExit(f"no per-step metrics found in {log_path}")
    return steps


def moving_average(values: list[float], window: int) -> list[float]:
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = parse_steps(args.log)
    ordered = sorted(steps)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for axis, (key, label, colour) in zip(axes.ravel(), METRICS):
        xs = [step for step in ordered if key in steps[step]]
        ys = [steps[step][key] for step in xs]
        if not ys:
            axis.set_visible(False)
            continue
        axis.plot(xs, ys, color=colour, alpha=0.35, linewidth=1, label="per step")
        axis.plot(
            xs,
            moving_average(ys, args.window),
            color=colour,
            linewidth=2,
            label=f"{args.window}-step mean",
        )
        axis.axhline(sum(ys) / len(ys), color="black", linestyle=":", linewidth=1, label="overall mean")
        axis.set_title(label, fontsize=11)
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")
    title = args.title or f"{args.log.name}: {len(ordered)} steps"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")
    for key, label, _ in METRICS:
        first = [steps[step][key] for step in ordered[:10] if key in steps[step]]
        last = [steps[step][key] for step in ordered[-10:] if key in steps[step]]
        if not first or not last:
            continue
        head = sum(first) / len(first)
        tail = sum(last) / len(last)
        change = 100 * (tail - head) / head if head else float("nan")
        print(f"{key:28s} first10={head:9.4g} last10={tail:9.4g} ({change:+.1f}%)")


if __name__ == "__main__":
    main()
