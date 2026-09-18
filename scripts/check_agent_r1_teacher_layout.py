#!/usr/bin/env python3
"""Probe Agent-R1's default OPD Teacher context and action-only mask layout."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("OPD Teacher layout output must be new")

    import torch
    from tensordict import TensorDict

    from agent_r1.agent_flow.agent_flow import AgentFlowManager
    from verl.protocol import DataProto

    prompt = [101, 102, 103]
    action_and_eos = [201, 202, 203]
    student_sequence = prompt + action_and_eos

    class FakeTeacher:
        teacher_key = "data_source"

        def __init__(self):
            self.requests = []

        async def compute_teacher_logprobs_single(self, *, sequence_ids, routing_key):
            self.requests.append({"sequence_ids": sequence_ids, "routing_key": routing_key})
            return (
                torch.tensor(sequence_ids, dtype=torch.int32),
                torch.zeros(len(sequence_ids), dtype=torch.float32),
            )

    async def probe(mask: list[int]) -> tuple[dict, list[dict]]:
        teacher = FakeTeacher()
        manager = object.__new__(AgentFlowManager)
        manager.config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout={"pad_token_id": 0})
        )
        manager.teacher_server_manager = teacher
        batch = TensorDict(
            {
                "prompts": torch.tensor([prompt]),
                "responses": torch.tensor([action_and_eos]),
                "input_ids": torch.tensor([student_sequence]),
                "attention_mask": torch.ones((1, len(student_sequence)), dtype=torch.int64),
                "response_mask": torch.tensor([mask]),
            },
            batch_size=1,
        )
        output = await manager._compute_teacher_logprobs(DataProto(batch=batch))
        result = {
            "student_sequence_width": len(student_sequence),
            "teacher_ids_width": output.batch["teacher_ids"].shape[1],
            "teacher_logprobs_width": output.batch["teacher_logprobs"].shape[1],
            "response_mask": mask,
        }
        return result, teacher.requests

    full_mask, full_requests = asyncio.run(probe([1, 1, 1]))
    action_only, action_requests = asyncio.run(probe([1, 1, 0]))
    if full_requests != action_requests or full_requests[0]["sequence_ids"] != student_sequence:
        raise AssertionError("Agent-R1 Teacher request unexpectedly changed between masks")
    payload = {
        "artifact": "nonconfirmatory_agent_r1_default_teacher_layout_probe",
        "training_performed": False,
        "teacher_requested_student_sequence": full_requests[0]["sequence_ids"] == student_sequence,
        "full_response_mask": full_mask,
        "action_only_eos_mask": action_only,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        "Agent-R1 default Teacher layout: "
        f"full={full_mask['teacher_ids_width']}, "
        f"action-only={action_only['teacher_ids_width']}, "
        f"Student={len(student_sequence)}"
    )
    print(args.output)


if __name__ == "__main__":
    main()
