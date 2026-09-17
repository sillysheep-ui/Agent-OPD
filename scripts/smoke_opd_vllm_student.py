#!/usr/bin/env python3
"""Inspect real vLLM Student termination and the pilot action mask, without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.io import read_jsonl, rollout_turn_from_dict
from omniopd.opd_adapter import extract_strict_action_tokens, prompts_for_opd_turn
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file
from omniopd.tokenization import apply_chat_template_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-pool", type=Path, required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    state_pool = args.state_pool.resolve()
    student_path = args.student_model.resolve()
    output_path = args.output.resolve()
    if (
        args.state_index < 0
        or args.seed < 0
        or args.max_new_tokens <= 0
        or output_path.exists()
        or student_path in output_path.parents
        or not state_pool.is_file()
        or not (student_path / "config.json").is_file()
    ):
        raise SystemExit("invalid Student vLLM smoke input or output")
    turn = None
    for index, row in enumerate(read_jsonl(state_pool)):
        if index == args.state_index:
            turn = rollout_turn_from_dict(row)
            break
    if turn is None:
        raise SystemExit("requested ALFWorld state is absent")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        str(student_path), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    student_prompt, _ = prompts_for_opd_turn(turn)
    prompt_ids = apply_chat_template_ids(
        tokenizer, student_prompt, add_generation_prompt=True, enable_thinking=False
    )
    student = LLM(
        model=str(student_path),
        tokenizer=str(student_path),
        trust_remote_code=False,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.35,
        max_model_len=2048,
        enforce_eager=True,
    )
    outputs = student.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            seed=args.seed,
        ),
        use_tqdm=False,
    )
    if len(outputs) != 1 or outputs[0].prompt_token_ids != prompt_ids or len(outputs[0].outputs) != 1:
        raise ValueError("vLLM returned an unexpected Student prompt or output count")
    candidate = outputs[0].outputs[0]
    generated_ids = [int(value) for value in candidate.token_ids]
    strict_valid = False
    strict_failure = None
    canonical_action = None
    try:
        _, _, canonical_action = extract_strict_action_tokens(
            tokenizer, generated_ids, turn.state.admissible_actions
        )
        strict_valid = True
    except ValueError as error:
        strict_failure = str(error)
    payload = {
        "artifact": "nonconfirmatory_vllm_student_termination_smoke",
        "training_performed": False,
        "code_revision": git_revision(ROOT),
        "code": fingerprint_code_tree(ROOT),
        "state_pool_sha256": sha256_file(state_pool),
        "state_hash": turn.state.state_hash,
        "student_config_sha256": sha256_file(student_path / "config.json"),
        "student_prompt_tokens": len(prompt_ids),
        "sampling": {"seed": args.seed, "temperature": 1.0, "max_new_tokens": args.max_new_tokens},
        "generated_token_ids": generated_ids,
        "generated_text": tokenizer.decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        "finish_reason": candidate.finish_reason,
        "stop_reason": candidate.stop_reason,
        "eos_token_id": tokenizer.eos_token_id,
        "ends_with_eos": bool(generated_ids and generated_ids[-1] == tokenizer.eos_token_id),
        "strict_action_valid": strict_valid,
        "strict_action_failure": strict_failure,
        "canonical_action": canonical_action,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        f"vLLM Student termination: EOS={payload['ends_with_eos']}, "
        f"strict_action_valid={strict_valid}, finish_reason={candidate.finish_reason}"
    )
    print(output_path)


if __name__ == "__main__":
    main()
