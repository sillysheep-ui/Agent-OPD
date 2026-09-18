#!/usr/bin/env python3
"""Probe whether Agent-R1's OPD tail slice respects an action-only loss mask.

This is a diagnostic, not a training or paper-result script.  Run it with the
pinned Agent-R1 veRL fork on PYTHONPATH; it never allocates a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing diagnostic")

    import torch
    from tensordict import TensorDict

    from verl.trainer.distillation.losses import _slice_nested_sequence_to_response_padding

    # Three prompt tokens, two action tokens, then EOS.  Scores are unique so
    # alignment errors cannot be hidden by equality or zero padding.
    scores = [0.0, 0.0, 0.0, 11.0, 12.0, 99.0]
    teacher = torch.nested.nested_tensor_from_jagged(
        torch.tensor(scores), offsets=torch.tensor([0, len(scores)])
    )
    cases = {}
    for name, mask in (("full_response", [1, 1, 1]), ("action_only", [1, 1, 0])):
        data = TensorDict({"response_mask": torch.tensor([mask], dtype=torch.bool)}, batch_size=[1])
        observed = _slice_nested_sequence_to_response_padding(teacher, data)[0].tolist()
        cases[name] = {"response_mask": mask, "observed_teacher_scores": observed}

    expected_action_scores = scores[3:5]
    payload = {
        "artifact": "nonconfirmatory_agent_r1_loss_alignment_probe",
        "training_performed": False,
        "source_sequence_scores": scores,
        "expected_action_scores": expected_action_scores,
        "cases": cases,
        "action_only_matches_action_scores": cases["action_only"]["observed_teacher_scores"]
        == expected_action_scores + [0.0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
