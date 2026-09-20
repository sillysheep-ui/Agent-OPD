#!/usr/bin/env python3
"""Compare a real action-format Student output with Agent-R1's tool-call parser."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--vllm-student-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not (args.student_model / "config.json").is_file():
        raise SystemExit("invalid Student format smoke input or output")
    sample = json.loads(args.vllm_student_smoke.read_text(encoding="utf-8"))
    if not sample.get("strict_action_valid") or not sample.get("generated_token_ids"):
        raise ValueError("source must be a verified, nonconfirmatory real Student action")

    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.tool_parser import ToolParser

    from omniopd.prompts import STUDENT_SYSTEM_PROMPT
    from recipes.alfworld.alfworld_agent_flow import _recover_tool_calls_from_text
    from recipes.alfworld.prompts import ALFWORLD_SYSTEM_PROMPT

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.student_model), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    token_ids = [int(value) for value in sample["generated_token_ids"]]
    response_text = tokenizer.decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    _, tool_calls = asyncio.run(
        ToolParser.get_tool_parser("hermes", tokenizer).extract_tool_calls(token_ids)
    )
    recovered = _recover_tool_calls_from_text(response_text)
    payload = {
        "artifact": "nonconfirmatory_agent_r1_student_action_format_check",
        "training_performed": False,
        "state_hash": sample["state_hash"],
        "student_strict_action_valid": True,
        "decoded_student_response": response_text,
        "agent_r1_hermes_tool_calls": len(tool_calls),
        "agent_r1_fallback_tool_calls": len(recovered),
        "same_system_prompt": STUDENT_SYSTEM_PROMPT == ALFWORLD_SYSTEM_PROMPT,
        "directly_compatible": bool(tool_calls or recovered)
        and STUDENT_SYSTEM_PROMPT == ALFWORLD_SYSTEM_PROMPT,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        "Agent-R1 direct Student format compatibility: "
        f"{payload['directly_compatible']} "
        f"(Hermes={len(tool_calls)}, fallback={len(recovered)})"
    )
    print(args.output)


if __name__ == "__main__":
    main()
