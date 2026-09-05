#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.prompts import STUDENT_SYSTEM_PROMPT
from omniopd.provenance import (
    fingerprint_code_tree,
    fingerprint_path,
    git_revision,
    sha256_file,
)
from omniopd.tokenization import apply_chat_template_ids


def _artifact_digest(value: dict) -> str | None:
    return value.get("sha256") or value.get("tree_sha256")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter in FP32 and verify logits equivalence"
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    base = Path(args.base_model)
    adapter = Path(args.adapter)
    training_manifest_path = Path(args.training_manifest)
    tokenizer_path = Path(args.tokenizer or args.base_model)
    output = Path(args.output_dir)
    if (
        not base.exists()
        or not adapter.exists()
        or not tokenizer_path.exists()
        or not training_manifest_path.is_file()
    ):
        raise SystemExit("base model, final adapter, tokenizer, and training manifest must exist")
    if output.exists():
        raise SystemExit(f"refusing to overwrite merge output: {output}")
    if args.atol < 0 or args.rtol < 0:
        raise SystemExit("--atol and --rtol must be non-negative")
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8")
    )
    try:
        final_step = int(training_manifest["training_contract"]["total_optimizer_steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"training manifest is incomplete: {error}") from error
    if (
        not current_revision
        or training_manifest.get("artifact") != "omniopd_training_launch"
        or training_manifest.get("protocol_version") != "omniopd-v1"
        or training_manifest.get("code") != current_code
        or training_manifest.get("code_revision") != current_revision
        or adapter.resolve()
        != (training_manifest_path.parent / f"global_step_{final_step}").resolve()
        or _artifact_digest(training_manifest.get("inputs", {}).get("model_path", {}))
        != _artifact_digest(fingerprint_path(base))
    ):
        raise SystemExit(
            "merge inputs must match the final adapter/base in a canonical training run"
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )
    messages = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Task:\ninspect the room\n\nObservation:\nYou are in a room.\n\n"
                "Admissible actions:\n- look"
            ),
        },
    ]
    input_ids = torch.tensor(
        apply_chat_template_ids(
            tokenizer,
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    adapter_model = PeftModel.from_pretrained(
        base_model, adapter, autocast_adapter_dtype=False
    ).eval()
    with torch.no_grad():
        reference = adapter_model(input_ids=input_ids, use_cache=False).logits.float().cpu()
    merged = adapter_model.merge_and_unload(safe_merge=True).eval()
    with torch.no_grad():
        merged_logits = merged(input_ids=input_ids, use_cache=False).logits.float().cpu()
    difference = (reference - merged_logits).abs()
    max_abs_error = float(difference.max().item())
    scale = float(reference.abs().max().item())
    max_relative_error = max_abs_error / max(scale, 1e-12)
    if not torch.allclose(reference, merged_logits, atol=args.atol, rtol=args.rtol):
        raise RuntimeError(
            "adapter and merged logits are not equivalent: "
            f"max_abs={max_abs_error:.6g}, max_relative={max_relative_error:.6g}"
        )

    model_output = output / "model"
    model_output.mkdir(parents=True)
    merged.save_pretrained(model_output, safe_serialization=True)
    tokenizer.save_pretrained(model_output)
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "fp32_merged_lora",
        "code": current_code,
        "code_revision": current_revision,
        "base_model": fingerprint_path(base),
        "adapter": fingerprint_path(adapter),
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "tokenizer": fingerprint_path(tokenizer_path),
        "merged_model": fingerprint_path(model_output),
        "merge_dtype": "fp32",
        "verification": {
            "enable_thinking": False,
            "atol": args.atol,
            "rtol": args.rtol,
            "max_abs_logit_error": max_abs_error,
            "max_relative_logit_error": max_relative_error,
            "passed": True,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
