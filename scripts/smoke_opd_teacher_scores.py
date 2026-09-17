#!/usr/bin/env python3
"""One-state, no-update Student-token/Teacher-score technical smoke test.

The input may be an old state pool. This script never calls its output a
confirmatory experiment, never trains, and never uses the pool's old Student
action as the OPD response.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.io import read_jsonl, rollout_turn_from_dict
from omniopd.opd_adapter import (
    align_teacher_sampled_token_logprobs,
    audit_shared_token_id_space,
    prompts_for_opd_turn,
)
from omniopd.parser import parse_action
from omniopd.provenance import fingerprint_code_tree, git_revision, sha256_file
from omniopd.tokenization import apply_chat_template_ids


def extract_strict_action_tokens(tokenizer, generated_ids, admissible_actions):
    """Require exactly one action line terminated by EOS; exclude EOS from loss."""

    ids = [int(token_id) for token_id in generated_ids]
    if len(ids) < 2 or ids[-1] != tokenizer.eos_token_id:
        raise ValueError("Student did not finish an action with its EOS token")
    content_ids = ids[:-1]
    if any(token_id in set(tokenizer.all_special_ids) for token_id in content_ids):
        raise ValueError("Student action contains a special/reasoning token")
    text = tokenizer.decode(
        content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if not re.fullmatch(r"Action:[^\r\n]+(?:\r?\n)?", text):
        raise ValueError(f"Student response is not exactly one action line: {text!r}")
    parsed = parse_action(text, admissible_actions)
    if not parsed.valid:
        raise ValueError(f"Student action is not admissible: {parsed.failure_reason}: {text!r}")
    return content_ids, text, parsed.canonical_action


def _sampled_logprobs(model, prompt_ids, response_ids, torch):
    ids = [*prompt_ids, *response_ids]
    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
        response_logits = logits[:, len(prompt_ids) - 1 : len(ids) - 1, :].float()
        target_ids = input_ids[:, len(prompt_ids) :]
        values = torch.log_softmax(response_logits, dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        )
        result = values.squeeze(0).squeeze(-1).cpu().tolist()
    if len(result) != len(response_ids) or any(not math.isfinite(value) for value in result):
        raise ValueError("sampled-token logprob shape or finiteness is invalid")
    return result


def _one_turn(path: Path, index: int):
    if index < 0:
        raise ValueError("--state-index must be non-negative")
    for current, row in enumerate(read_jsonl(path)):
        if current == index:
            return rollout_turn_from_dict(row)
    raise ValueError(f"state index {index} is absent from {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-pool", type=Path, required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    state_pool = args.state_pool.resolve()
    student_path = args.student_model.resolve()
    teacher_path = args.teacher_model.resolve()
    output = args.output.resolve()
    if (
        output.exists()
        or output == state_pool
        or student_path in output.parents
        or teacher_path in output.parents
        or args.max_new_tokens <= 0
        or args.seed < 0
    ):
        raise SystemExit("invalid smoke output, seed, or generation limit")
    for path in (state_pool, student_path / "config.json", teacher_path / "config.json"):
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    turn = _one_turn(state_pool, args.state_index)
    student_prompt, teacher_prompt = prompts_for_opd_turn(turn)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("single-GPU CUDA is required for this technical smoke")
    student_tokenizer = AutoTokenizer.from_pretrained(
        str(student_path), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        str(teacher_path), use_fast=True, local_files_only=True, trust_remote_code=False
    )
    token_space = audit_shared_token_id_space(student_tokenizer, teacher_tokenizer)
    student_prompt_ids = apply_chat_template_ids(
        student_tokenizer,
        student_prompt,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    teacher_prompt_ids = apply_chat_template_ids(
        teacher_tokenizer,
        teacher_prompt,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not student_prompt_ids or not teacher_prompt_ids:
        raise ValueError("Student or Teacher prompt is empty")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.monotonic()
    student_model = AutoModelForCausalLM.from_pretrained(
        str(student_path), torch_dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=False, attn_implementation="sdpa",
    ).to("cuda:0").eval()
    student_input = torch.tensor([student_prompt_ids], dtype=torch.long, device="cuda:0")
    with torch.inference_mode():
        generated = student_model.generate(
            input_ids=student_input,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=student_tokenizer.eos_token_id,
            pad_token_id=student_tokenizer.pad_token_id,
        )
    generated_ids = generated[0, len(student_prompt_ids) :].tolist()
    content_ids, action_text, canonical_action = extract_strict_action_tokens(
        student_tokenizer, generated_ids, turn.state.admissible_actions
    )
    student_logprobs = _sampled_logprobs(
        student_model, student_prompt_ids, content_ids, torch
    )
    student_seconds = time.monotonic() - started
    del student_model, student_input, generated
    gc.collect()
    torch.cuda.empty_cache()

    started = time.monotonic()
    teacher_model = AutoModelForCausalLM.from_pretrained(
        str(teacher_path), torch_dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=False, attn_implementation="sdpa",
    ).to("cuda:0").eval()
    teacher_scores = _sampled_logprobs(
        teacher_model, teacher_prompt_ids, content_ids, torch
    )
    aligned_scores = align_teacher_sampled_token_logprobs(
        teacher_prompt_ids=teacher_prompt_ids,
        student_response_ids=content_ids,
        scored_sequence_ids=[*teacher_prompt_ids, *content_ids],
        scored_token_logprobs=[None] * len(teacher_prompt_ids) + teacher_scores,
    )
    teacher_seconds = time.monotonic() - started
    payload = {
        "artifact": "nonconfirmatory_single_state_token_opd_forward_smoke",
        "training_performed": False,
        "code_revision": git_revision(ROOT),
        "code": fingerprint_code_tree(ROOT),
        "state_pool_sha256": sha256_file(state_pool),
        "state_hash": turn.state.state_hash,
        "game_id": turn.state.game_id,
        "turn_index": turn.state.turn_index,
        "student_config_sha256": sha256_file(student_path / "config.json"),
        "teacher_config_sha256": sha256_file(teacher_path / "config.json"),
        "token_id_space": token_space,
        "sampling": {"seed": args.seed, "temperature": 1.0, "max_new_tokens": args.max_new_tokens},
        "student_prompt_tokens": len(student_prompt_ids),
        "teacher_prompt_tokens": len(teacher_prompt_ids),
        "student_generated_tokens_including_eos": len(generated_ids),
        "action_only_mask": [1] * len(content_ids) + [0],
        "student_action_token_ids": content_ids,
        "student_action_text": action_text,
        "canonical_action": canonical_action,
        "student_sampled_token_logprobs": student_logprobs,
        "teacher_sampled_token_logprobs": aligned_scores,
        "student_seconds": student_seconds,
        "teacher_seconds": teacher_seconds,
        "teacher_scored_tokens": len(aligned_scores),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Student-generated action Teacher-scored without update: {output}")


if __name__ == "__main__":
    main()
