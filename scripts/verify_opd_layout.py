#!/usr/bin/env python3
"""Verify the token-OPD layout contract offline, without a training run.

veRL concatenates response tensors across a batch, so every sample must
produce a response of the batch's fixed width and a Teacher row array whose
length equals the Student prompt plus that response. Each unforeseen edge
case previously surfaced as a crash only when a batch happened to contain it,
so the cases are enumerated here instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from omniopd.opd_adapter import (  # noqa: E402
    align_teacher_rows_to_student_layout,
    extract_strict_action_tokens,
    response_mask_for_action_span,
)

CASES = [
    ("action_with_terminator", "Action: look", True),
    ("action_then_hallucinated_observation", "Action: go to sofa 1\nObservation: On the sofa 1, you see a cat.", True),
    ("action_only_no_terminator", "Action: look", False),
    ("rambling_before_action", "Thought: I should look around first.\nAction: look", True),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--response-length", type=int, default=512)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    admissible = ["look", "go to sofa 1", "go to countertop 1"]
    failures = []

    for name, text, add_terminator in CASES:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if add_terminator:
            ids = [*ids, tokenizer.convert_tokens_to_ids("<|im_end|>")]
        # veRL pads the response to the batch width with pad ids.
        padded = ids + [tokenizer.pad_token_id] * (args.response_length - len(ids))
        span, _, _ = extract_strict_action_tokens(
            tokenizer, padded, admissible, require_admissible=False
        )
        mask = response_mask_for_action_span(len(padded), len(span))
        rows, _ = align_teacher_rows_to_student_layout(
            student_prompt_length=100,
            student_response_ids=padded,
            teacher_rows=[[0.0]] * (100 + len(span)),
            teacher_ids=[[0]] * (100 + len(span)),
            pad_token_id=tokenizer.pad_token_id,
        )
        invariants = {
            "mask_len": len(mask) == len(padded),
            "mask_sum_equals_span": sum(mask) == len(span),
            "teacher_rows_width": len(rows) == 100 + len(padded),
        }
        status = "OK " if all(invariants.values()) else "FAIL"
        if status == "FAIL":
            failures.append((name, invariants, len(span), len(padded), text[:60]))
        print(f"{status} {name:38s} span={len(span):3d} response={len(padded):3d} {invariants}")

    # Failure modes that must raise rather than silently mis-align.
    for name, text in (
        ("no_action_line", "Thought: I am thinking."),
        ("empty_response", ""),
    ):
        padded = tokenizer(text, add_special_tokens=False)["input_ids"] or [tokenizer.pad_token_id]
        padded = padded + [tokenizer.pad_token_id] * max(0, 8 - len(padded))
        try:
            extract_strict_action_tokens(
                tokenizer, padded, admissible, require_admissible=False
            )
        except ValueError:
            print(f"OK  {name:38s} correctly raised")
        else:
            failures.append((name, "did not raise", 0, 0, text))
            print(f"FAIL {name:38s} did not raise")

    if failures:
        print("\nFAILURES:", failures)
        raise SystemExit(1)
    print("\nlayout contract: OK")


if __name__ == "__main__":
    main()
