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
    describe_supervision_span,
    extract_strict_action_tokens,
    response_mask_for_action_span,
)

# name, emitted text, terminated by <|im_end|>, expected supervision mode
CASES = [
    ("action_with_terminator", "Action: look", True, "action_line"),
    (
        "action_then_hallucinated_observation",
        "Action: go to sofa 1\nObservation: On the sofa 1, you see a cat.",
        True,
        "action_line",
    ),
    ("action_only_no_terminator", "Action: look", False, "action_line"),
    ("rambling_before_action", "Thought: I should look around first.\nAction: look", True, "action_line"),
    (
        "give_up_without_action_line",
        "Thought: the only object manipulated was a bowl.\n"
        "No further actions are necessary. The task cannot be completed as written.",
        True,
        "whole_response",
    ),
    (
        "give_up_without_action_line_or_terminator",
        "The task cannot be completed as written.",
        False,
        "whole_response",
    ),
    # A turn that emitted nothing but its terminator keeps that one token: the
    # Teacher scorer has no other sequence to score.
    ("terminator_only", "", True, "whole_response"),
    ("response_without_any_content", "", False, "empty"),
]

# A turn whose content is a bare terminator carries no supervision. It must
# return an empty span (zero-mask sample) rather than aborting the run.
STRICT_CASES = [
    ("strict_no_action_line", "Thought: I am thinking."),
    ("strict_empty_response", ""),
]


def emitted_only(padded_ids, tokenizer):
    """Independently recompute the emitted tokens of a padded response.

    Mirrors the frozen contract: drop right padding, drop the trailing
    terminator, but never reduce a non-empty response to nothing.
    """

    ids = list(padded_ids)
    while ids and ids[-1] == tokenizer.pad_token_id:
        ids.pop()
    special_ids = set(tokenizer.all_special_ids)
    while len(ids) > 1 and ids[-1] in special_ids:
        ids.pop()
    return ids


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

    for name, text, add_terminator, expected_mode in CASES:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if add_terminator:
            ids = [*ids, tokenizer.convert_tokens_to_ids("<|im_end|>")]
        # veRL pads the response to the batch width with pad ids.
        padded = ids + [tokenizer.pad_token_id] * (args.response_length - len(ids))
        span, span_text, _ = extract_strict_action_tokens(
            tokenizer, padded, admissible, require_admissible=False
        )
        mode = describe_supervision_span(span, span_text)
        mask = response_mask_for_action_span(len(padded), len(span))
        emitted = emitted_only(padded, tokenizer)
        if expected_mode == "action_line":
            # An action turn supervises a prefix of its emitted tokens and
            # stops before any hallucinated future or padding.
            span_is_prefix = bool(span) and span == emitted[: len(span)]
        else:
            # A turn without an action line supervises exactly what it emitted.
            span_is_prefix = span == emitted
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
            "mode": mode == expected_mode,
            "span_matches_emitted": span_is_prefix,
        }
        status = "OK " if all(invariants.values()) else "FAIL"
        if status == "FAIL":
            failures.append((name, invariants, len(span), len(padded), text[:60]))
        print(
            f"{status} {name:38s} mode={mode:13s} span={len(span):3d} "
            f"response={len(padded):3d} {invariants}"
        )

    # The fail-closed policy still applies to the teacher-sampling arm, where a
    # turn without an admissible action line must stop the run.
    for name, text in STRICT_CASES:
        padded = tokenizer(text, add_special_tokens=False)["input_ids"] or [tokenizer.pad_token_id]
        padded = padded + [tokenizer.pad_token_id] * max(0, 8 - len(padded))
        try:
            extract_strict_action_tokens(
                tokenizer, padded, admissible, require_admissible=True
            )
        except ValueError:
            print(f"OK  {name:38s} correctly raised under require_admissible=True")
        else:
            failures.append((name, "did not raise", 0, 0, text))
            print(f"FAIL {name:38s} did not raise under require_admissible=True")

    if failures:
        print("\nFAILURES:", failures)
        raise SystemExit(1)
    print("\nlayout contract: OK")


if __name__ == "__main__":
    main()
