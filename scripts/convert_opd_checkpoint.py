#!/usr/bin/env python3
"""Convert a veRL FSDP LoRA checkpoint into a vLLM-loadable PEFT adapter.

veRL's PPO trainer stores sharded FSDP tensors, while vLLM's `--enable-lora`
expects a PEFT directory (adapter_config.json + adapter_model.safetensors).
If veRL already wrote PEFT files this script just reports them; otherwise it
extracts the LoRA A/B pairs from the shard and writes the PEFT layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGET_MODULE_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


def find_existing_adapter(root: Path) -> Path | None:
    for candidate in sorted(root.rglob("adapter_config.json")):
        if (candidate.parent / "adapter_model.safetensors").is_file():
            return candidate.parent
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint directory is missing: {checkpoint}")

    existing = find_existing_adapter(checkpoint)
    if existing is not None:
        print(f"PEFT adapter already present: {existing}")
        return

    import torch
    from safetensors.torch import save_file

    actor = checkpoint / "actor"
    shards = sorted(actor.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        raise SystemExit(f"no FSDP model shard found under {actor}")

    state: dict[str, object] = {}
    for shard in shards:
        loaded = torch.load(shard, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "model" in loaded and isinstance(loaded["model"], dict):
            loaded = loaded["model"]
        if not isinstance(loaded, dict):
            raise SystemExit(f"unexpected shard payload type: {type(loaded)!r}")
        for key, value in loaded.items():
            state[key] = value
    print(f"loaded {len(state)} tensors from {len(shards)} shard(s)")

    def peft_name(raw: str) -> str | None:
        name = raw
        for prefix in ("_fsdp_wrapped_module.", "module.", "_orig_mod."):
            name = name.replace(prefix, "")
        if ".lora_A." in name:
            base, _ = name.split(".lora_A.", 1)
            return f"{base}.lora_A.weight"
        if ".lora_B." in name:
            base, _ = name.split(".lora_B.", 1)
            return f"{base}.lora_B.weight"
        return None

    adapter: dict[str, object] = {}
    for key, value in state.items():
        name = peft_name(key)
        if name is None:
            continue
        if not hasattr(value, "detach"):
            continue
        adapter[name] = value.detach().to(torch.float32).contiguous()

    if not adapter:
        sample = list(state)[:6]
        raise SystemExit(
            "no LoRA A/B tensors found in the shard; sample keys: " + ", ".join(map(str, sample))
        )
    modules = sorted(
        {
            name.split(".lora_")[0].split(".")[-1]
            for name in adapter
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    save_file(adapter, str(output / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": args.rank,
        "lora_alpha": args.alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": modules or list(TARGET_MODULE_SUFFIXES),
        "base_model_name_or_path": "base",
    }
    (output / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote PEFT adapter with {len(adapter)} tensors to {output}")
    print("target modules:", config["target_modules"])


if __name__ == "__main__":
    main()
