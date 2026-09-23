#!/usr/bin/env python3
"""Merge an OPD LoRA adapter into its base model and prove the merge is faithful.

Serving a merged checkpoint avoids depending on the vLLM LoRA path, but a merge
that silently drops the adapter would evaluate the base model under a trained
name. This script therefore measures three sets of logits on one prompt in
float32 before it writes anything:

* base            -- the frozen Student,
* adapter-applied -- the same weights with the adapter active at runtime,
* merged          -- the exported checkpoint, reloaded.

It fails when the adapter has no effect at all, and reports the merge gap so a
low-precision export that distorts the adapter cannot pass unnoticed. Run it on
CPU: it needs no GPU and no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PROMPT = "You are an ALFWorld household agent.\nTask: put a clean cup in microwave."


def last_logits(model, tokenizer, prompt: str):
    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        return model(**encoded).logits[0, -1].float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--export-dtype", default="bfloat16")
    parser.add_argument("--max-merge-gap-ratio", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    base_dir = args.base_model.resolve()
    adapter_dir = args.adapter.resolve()
    output = args.output_dir.resolve()
    if not (adapter_dir / "adapter_config.json").is_file():
        raise SystemExit(f"adapter directory is incomplete: {adapter_dir}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite a non-empty output directory: {output}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(base_dir), torch_dtype=torch.float32, local_files_only=True
    ).to(device)
    logits_base = last_logits(base, tokenizer, args.prompt)

    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    logits_adapter = last_logits(peft_model, tokenizer, args.prompt)
    adapter_effect = float((logits_adapter - logits_base).abs().max())
    if adapter_effect == 0.0:
        raise SystemExit(
            "the adapter has no effect on the model logits; refusing to export it as a "
            "trained checkpoint"
        )

    merged = peft_model.merge_and_unload()
    export_dtype = getattr(torch, args.export_dtype)
    merged = merged.to(export_dtype)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    del merged, peft_model, base

    reloaded = AutoModelForCausalLM.from_pretrained(
        str(output), torch_dtype=export_dtype, local_files_only=True
    ).to(device)
    logits_merged = last_logits(reloaded, tokenizer, args.prompt)
    merge_gap = float((logits_merged - logits_adapter).abs().max())
    scale = float(logits_base.abs().max())
    ratio = merge_gap / scale if scale else float("inf")
    payload = {
        "artifact": "opd_adapter_merge",
        "base_model": str(base_dir),
        "adapter": str(adapter_dir),
        "merged_model": str(output),
        "export_dtype": args.export_dtype,
        "adapter_effect_max_abs": adapter_effect,
        "merge_gap_max_abs": merge_gap,
        "logit_scale_max_abs": scale,
        "merge_gap_ratio": ratio,
        "max_merge_gap_ratio": args.max_merge_gap_ratio,
    }
    (output / "merge_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if ratio > args.max_merge_gap_ratio:
        raise SystemExit(
            f"merge gap {merge_gap} is {ratio:.3f} of the logit scale {scale}, above the "
            f"allowed {args.max_merge_gap_ratio}; the exported checkpoint is not the adapter"
        )
    print("merge verified")


if __name__ == "__main__":
    main()
