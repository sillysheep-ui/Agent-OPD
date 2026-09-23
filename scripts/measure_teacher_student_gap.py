#!/usr/bin/env python3
"""Measure the teacher-student logprob gap on real student actions.

Every sampled-token distillation estimator's usefulness is decided by the
distribution of (student_logprob - teacher_logprob) on the tokens the run would
actually supervise. This script measures that distribution on recorded student
actions, and reports the two things that decide which estimator is even legal:

* the share of tokens where the k3 estimator's value hits its own clamp and
  therefore contributes exactly zero gradient (its value is clamped to +/-10,
  which happens on BOTH sides of the gap: teacher-minus-student >= ~2.6 and
  student-minus-teacher >= ~11), and
* how often the teacher's argmax token differs from the action token the student
  actually produced, i.e. how often supervision carries a positive target.

It runs both models in bfloat16 on one GPU and needs no environment rollout.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def action_text(turn) -> str | None:
    """The supervised continuation: the first action line the Student emitted."""

    executed = turn["student"].get("executed_action")
    if executed:
        return f"Action: {executed}"
    raw = str(turn["student"].get("raw") or "")
    for line in raw.splitlines():
        if line.strip().lower().startswith("action:"):
            return line.strip()
    return None


def load_turns(pool_path: Path, limit: int, seed: int) -> list[dict]:
    turns = []
    for line in pool_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        for turn in episode.get("turns", []):
            text = action_text(turn)
            if text:
                turns.append(
                    {
                        "task_type": episode.get("task_type") or turn["state"].get("task_type"),
                        "messages": turn["state"]["messages"],
                        "action": text,
                    }
                )
    if not turns:
        raise SystemExit(f"no usable turns in {pool_path}")
    random.Random(seed).shuffle(turns)
    # One pass per task type first, so the sample is not dominated by one family.
    by_task: dict[str, list[dict]] = defaultdict(list)
    for turn in turns:
        by_task[turn["task_type"]].append(turn)
    selected: list[dict] = []
    while len(selected) < limit and any(by_task.values()):
        for task in sorted(by_task):
            if by_task[task] and len(selected) < limit:
                selected.append(by_task[task].pop())
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0")
    turns = load_turns(args.pool, args.limit, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(str(args.student), local_files_only=True)
    records = []
    for name, path in (("student", args.student), ("teacher", args.teacher)):
        model = AutoModelForCausalLM.from_pretrained(
            str(path), dtype=torch.bfloat16, local_files_only=True
        ).to(device)
        model.eval()
        for turn in turns:
            prompt_ids = tokenizer.apply_chat_template(
                turn["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            action_ids = tokenizer(turn["action"], add_special_tokens=False)["input_ids"]
            ids = torch.tensor([prompt_ids + action_ids], device=device)
            with torch.no_grad():
                logits = model(ids).logits[0]
            start = len(prompt_ids) - 1
            span = logits[start : start + len(action_ids)]
            logprobs = torch.log_softmax(span.float(), dim=-1)
            targets = torch.tensor(action_ids, device=device)
            picked = logprobs.gather(1, targets[:, None]).squeeze(1)
            argmax_ids = span.argmax(dim=-1).tolist()
            turn.setdefault("measurements", {})[name] = {
                "logprobs": [float(value) for value in picked],
                "argmax_first": int(argmax_ids[0]),
                "argmax_text_first": tokenizer.decode([int(argmax_ids[0])], skip_special_tokens=False),
                "argmax_ids": argmax_ids,
                "action_ids": action_ids,
            }
        del model
        torch.cuda.empty_cache()

    gaps = []
    dead = 0
    positive_target = 0
    first_token_gaps = []
    rest_gaps = []
    first_argmax_tokens: dict[str, int] = defaultdict(int)
    first_token_clamped = 0
    per_task: dict[str, list[float]] = defaultdict(list)
    for turn in turns:
        student = turn["measurements"]["student"]
        teacher = turn["measurements"]["teacher"]
        for position, (s, t) in enumerate(zip(student["logprobs"], teacher["logprobs"])):
            gap = s - t
            gaps.append(gap)
            (first_token_gaps if position == 0 else rest_gaps).append(gap)
            per_task[turn["task_type"]].append(gap)
            if -gap >= 2.6:  # teacher - student >= 2.6 -> k3 value saturates at 10
                dead += 1
        first_argmax_tokens[str(teacher["argmax_text_first"])] += 1
        if student["argmax_first"] != teacher["argmax_first"]:
            positive_target += 1

    # Reproduce k3's own arithmetic to find which tokens it can actually push.
    k3_dead = 0
    for turn in turns:
        for s, t in zip(
            turn["measurements"]["student"]["logprobs"],
            turn["measurements"]["teacher"]["logprobs"],
        ):
            kl = min(20.0, max(-20.0, t - s))
            kld = math.exp(kl) - kl - 1
            if abs(min(10.0, max(-10.0, kld)) - kld) > 1e-6:
                k3_dead += 1
    for turn in turns:
        s = turn["measurements"]["student"]["logprobs"][0]
        t = turn["measurements"]["teacher"]["logprobs"][0]
        kl = min(20.0, max(-20.0, t - s))
        kld = math.exp(kl) - kl - 1
        if abs(min(10.0, max(-10.0, kld)) - kld) > 1e-6:
            first_token_clamped += 1

    gaps_sorted = sorted(gaps)

    def percentile(p: float) -> float:
        index = min(len(gaps_sorted) - 1, int(p * len(gaps_sorted)))
        return gaps_sorted[index]

    payload = {
        "artifact": "teacher_student_gap_measurement",
        "pool": str(args.pool),
        "student": str(args.student),
        "teacher": str(args.teacher),
        "turns": len(turns),
        "tokens": len(gaps),
        "gap_mean": sum(gaps) / len(gaps),
        "gap_median": percentile(0.5),
        "gap_rms": (sum(gap * gap for gap in gaps) / len(gaps)) ** 0.5,
        "gap_p90": percentile(0.9),
        "gap_p99": percentile(0.99),
        "gap_max": gaps_sorted[-1],
        "gap_min": gaps_sorted[0],
        "share_with_gap_ge_2.6": dead / len(gaps),
        "share_k3_value_clamped": k3_dead / len(gaps),
        "first_token_gap_mean": (sum(first_token_gaps) / len(first_token_gaps)) if first_token_gaps else None,
        "first_token_gap_median": sorted(first_token_gaps)[len(first_token_gaps) // 2] if first_token_gaps else None,
        "rest_gap_mean": (sum(rest_gaps) / len(rest_gaps)) if rest_gaps else None,
        "rest_gap_median": sorted(rest_gaps)[len(rest_gaps) // 2] if rest_gaps else None,
        "share_k3_clamped_at_first_token": first_token_clamped / len(turns) if turns else None,
        "teacher_argmax_text_at_first_token": dict(
            sorted(first_argmax_tokens.items(), key=lambda item: -item[1])[:6]
        ),
        "teacher_argmax_differs_at_first_token": positive_target / len(turns),
        "gap_mean_by_task": {
            task: sum(values) / len(values) for task, values in sorted(per_task.items())
        },
    }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
