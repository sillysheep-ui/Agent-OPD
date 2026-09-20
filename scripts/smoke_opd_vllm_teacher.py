#!/usr/bin/env python3
"""Compare vLLM prompt_logprobs with a recorded HF Teacher forward smoke."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.io import read_jsonl, rollout_turn_from_dict
from omniopd.opd_adapter import prompts_for_opd_turn
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file
from omniopd.tokenization import apply_chat_template_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-pool", type=Path, required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--hf-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state_pool = args.state_pool.resolve()
    teacher_path = args.teacher_model.resolve()
    hf_smoke_path = args.hf_smoke.resolve()
    output_path = args.output.resolve()
    if (
        args.state_index < 0
        or output_path.exists()
        or teacher_path in output_path.parents
        or not state_pool.is_file()
        or not hf_smoke_path.is_file()
        or not (teacher_path / "config.json").is_file()
    ):
        raise SystemExit("invalid vLLM comparison input or output")
    turn = None
    for index, row in enumerate(read_jsonl(state_pool)):
        if index == args.state_index:
            turn = rollout_turn_from_dict(row)
            break
    if turn is None:
        raise SystemExit("requested ALFWorld state is absent")
    hf_smoke = json.loads(hf_smoke_path.read_text(encoding="utf-8"))
    if (
        hf_smoke.get("artifact") != "nonconfirmatory_single_state_token_opd_forward_smoke"
        or hf_smoke.get("state_hash") != turn.state.state_hash
        or hf_smoke.get("state_pool_sha256") != sha256_file(state_pool)
        or hf_smoke.get("teacher_config_sha256") != sha256_file(teacher_path / "config.json")
    ):
        raise SystemExit("HF smoke is not bound to this exact state and Teacher")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs

    tokenizer = AutoTokenizer.from_pretrained(
        str(teacher_path), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    _, teacher_prompt = prompts_for_opd_turn(turn)
    teacher_prompt_ids = apply_chat_template_ids(
        tokenizer, teacher_prompt, add_generation_prompt=True, enable_thinking=False
    )
    response_ids = [int(token) for token in hf_smoke["student_action_token_ids"]]
    if not teacher_prompt_ids or not response_ids:
        raise SystemExit("Teacher prompt or Student action is empty")
    sequence_ids = teacher_prompt_ids + response_ids
    started = time.monotonic()
    teacher = LLM(
        model=str(teacher_path),
        tokenizer=str(teacher_path),
        trust_remote_code=False,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.55,
        max_model_len=2048,
        enforce_eager=True,
    )
    outputs = teacher.generate(
        [{"prompt_token_ids": sequence_ids}],
        SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0),
        use_tqdm=False,
    )
    if len(outputs) != 1 or outputs[0].prompt_token_ids != sequence_ids:
        raise ValueError("vLLM returned a different Teacher input sequence")
    prompt_logprobs = outputs[0].prompt_logprobs
    if prompt_logprobs is None or len(prompt_logprobs) != len(sequence_ids):
        raise ValueError("vLLM prompt_logprobs length does not match the scored sequence")
    vllm_scores = []
    for position, token_id in enumerate(response_ids, len(teacher_prompt_ids)):
        item = prompt_logprobs[position]
        if item is None or token_id not in item:
            raise ValueError(f"vLLM omitted Student token ID at response position {position}")
        score = float(item[token_id].logprob)
        if not math.isfinite(score):
            raise ValueError(f"vLLM returned a non-finite score at position {position}")
        vllm_scores.append(score)
    hf_scores = [float(value) for value in hf_smoke["teacher_sampled_token_logprobs"]]
    if len(hf_scores) != len(vllm_scores):
        raise ValueError("HF/vLLM Student action token counts differ")
    verl_fields: dict[str, list] = {}
    extract_prompt_logprobs(outputs[0], 0, verl_fields)
    shifted_ids = [row[0] for row in verl_fields["prompt_ids"]]
    shifted_scores = [row[0] for row in verl_fields["prompt_logprobs"]]
    if shifted_ids != sequence_ids[1:] + [0] or len(shifted_scores) != len(sequence_ids):
        raise ValueError("real vLLM output does not satisfy veRL's shifted Teacher contract")
    verl_response_scores = shifted_scores[
        len(teacher_prompt_ids) - 1 : len(teacher_prompt_ids) - 1 + len(response_ids)
    ]
    if verl_response_scores != vllm_scores:
        raise ValueError("veRL extracted different Student-token scores from real vLLM output")
    differences = [abs(a - b) for a, b in zip(hf_scores, vllm_scores)]
    payload = {
        "artifact": "nonconfirmatory_opd_vllm_vs_hf_teacher_scores",
        "training_performed": False,
        "code_revision": git_revision(ROOT),
        "code": fingerprint_code_tree(ROOT),
        "state_hash": turn.state.state_hash,
        "hf_smoke_sha256": sha256_file(hf_smoke_path),
        "teacher_config_sha256": sha256_file(teacher_path / "config.json"),
        "teacher_prompt_tokens": len(teacher_prompt_ids),
        "student_action_token_ids": response_ids,
        "hf_teacher_logprobs": hf_scores,
        "vllm_teacher_logprobs": vllm_scores,
        "verl_extractor_shifted_contract_verified": True,
        "absolute_differences": differences,
        "max_absolute_difference": max(differences),
        "vllm_model_load_and_score_seconds": time.monotonic() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"vLLM/HF sampled-token comparison: {output_path}")


if __name__ == "__main__":
    main()
