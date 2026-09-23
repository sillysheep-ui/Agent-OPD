#!/usr/bin/env python3
"""Convert a veRL FSDP LoRA checkpoint into a vLLM-loadable PEFT adapter.

veRL stores the actor's LoRA parameters as DTensor shards: every rank file holds
one local piece of each trained tensor with `Shard(dim)` placement on a mesh of
size world_size. A single rank therefore cannot produce a full adapter, and the
pieces arrive wrapped in DTensors whose local storage safetensors refuses to
serialize. This script concatenates the local pieces in rank order, checks the
result against the placement metadata, and optionally checks every adapter
matrix against the base model's layer shapes.

The checks are the point: a silent mix-up here would train a model and then
evaluate a different one.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TARGET_MODULE_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
SHARD_PATTERN = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt\Z")
BASE_KEY_PREFIXES = ("base_model.model.", "base_model.")


def find_existing_adapter(root: Path) -> Path | None:
    for candidate in sorted(root.rglob("adapter_config.json")):
        if (candidate.parent / "adapter_model.safetensors").is_file():
            return candidate.parent
    return None


def rank_sort_key(path: Path) -> tuple[int, int]:
    """Order shard files by (world_size, rank), not by their names as strings."""

    match = SHARD_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"unexpected shard name: {path.name}")
    return (int(match.group(1)), int(match.group(2)))


def peft_name(raw: str) -> str | None:
    """Map a veRL parameter name onto the key PEFT writes for one adapter."""

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


def placement_dim(value) -> int | None:
    """Return the shard dimension of a DTensor-like value, or None if replicated."""

    placements = getattr(value, "placements", None)
    if not placements:
        return None
    dim = getattr(placements[0], "dim", None)
    return None if dim is None else int(dim)


def concatenated_shape(local_shapes, dim: int) -> tuple[int, ...]:
    """Shape of a DTensor rebuilt by concatenating its local pieces on ``dim``."""

    shapes = [tuple(int(size) for size in shape) for shape in local_shapes]
    if not shapes:
        raise ValueError("no local shards to concatenate")
    if dim < 0 or dim >= len(shapes[0]):
        raise ValueError(f"shard dimension {dim} is out of range for {shapes[0]}")
    for shape in shapes[1:]:
        if len(shape) != len(shapes[0]):
            raise ValueError("shards disagree on rank")
        for axis, (left, right) in enumerate(zip(shapes[0], shape)):
            if axis != dim and left != right:
                raise ValueError(f"shards disagree on axis {axis}: {left} != {right}")
    total = list(shapes[0])
    total[dim] = sum(shape[dim] for shape in shapes)
    return tuple(total)


def base_weight_key(adapter_key: str) -> str:
    """Turn ``base_model.model.model.layers.0...q_proj.lora_A.weight`` into the base key."""

    for prefix in BASE_KEY_PREFIXES:
        if adapter_key.startswith(prefix):
            return adapter_key[len(prefix):].rsplit(".lora_", 1)[0] + ".weight"
    raise ValueError(f"adapter key has no known base prefix: {adapter_key}")


def base_layer_shapes(base_dir: Path) -> dict[str, tuple[int, ...]]:
    """Read base weight shapes from safetensors headers without loading weights."""

    from safetensors import safe_open

    files = sorted(base_dir.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors shards under {base_dir}")
    shapes: dict[str, tuple[int, ...]] = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shapes[key] = tuple(int(size) for size in handle.get_slice(key).get_shape())
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=None)
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
    shards = sorted(actor.glob("model_world_size_*_rank_*.pt"), key=rank_sort_key)
    if not shards:
        raise SystemExit(f"no FSDP model shard found under {actor}")
    world_sizes = {rank_sort_key(path)[0] for path in shards}
    if len(world_sizes) != 1:
        raise SystemExit(f"shards disagree on world size: {sorted(world_sizes)}")
    ranks = sorted(rank_sort_key(path)[1] for path in shards)
    if ranks != list(range(len(shards))):
        raise SystemExit(f"shard ranks are not 0..{len(shards) - 1}: {ranks}")

    loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in shards]
    global_shapes: dict[str, tuple[int, ...]] = {}
    adapter: dict[str, torch.Tensor] = {}
    for key in loaded[0]:
        name = peft_name(str(key))
        if name is None:
            continue
        values = [state[key] for state in loaded]
        local = [value.to_local() if hasattr(value, "to_local") else value for value in values]
        dim = placement_dim(values[0])
        if dim is None:
            for other in local[1:]:
                if not torch.equal(local[0], other):
                    raise SystemExit(f"replicated tensor disagrees between ranks: {key}")
            tensor = local[0]
        else:
            expected = concatenated_shape([tensor.shape for tensor in local], dim)
            tensor = torch.cat(local, dim=dim)
            if tuple(tensor.shape) != expected:
                raise SystemExit(f"rebuilt shape {tuple(tensor.shape)} != expected {expected}")
        stated = getattr(values[0], "shape", None)
        if stated is not None and tuple(int(size) for size in stated) != tuple(tensor.shape):
            raise SystemExit(
                f"rebuilt {name} has shape {tuple(tensor.shape)} but the shard declared "
                f"{tuple(int(size) for size in stated)}"
            )
        global_shapes[name] = tuple(tensor.shape)
        adapter[name] = tensor.detach().to(torch.float32).contiguous().clone()

    if not adapter:
        sample = list(loaded[0])[:6]
        raise SystemExit(
            "no LoRA A/B tensors found in the shard; sample keys: " + ", ".join(map(str, sample))
        )

    paired = {}
    for name in adapter:
        base, kind = name.rsplit(".lora_", 1)
        paired.setdefault(base, {})[kind.split(".")[0]] = name
    incomplete = [base for base, parts in paired.items() if {"A", "B"} - set(parts)]
    if incomplete:
        raise SystemExit(f"adapter modules without both A and B: {incomplete[:5]}")
    for base, parts in paired.items():
        a_shape = adapter[parts["A"]].shape
        b_shape = adapter[parts["B"]].shape
        if a_shape[0] != args.rank or b_shape[1] != args.rank:
            raise SystemExit(
                f"LoRA rank mismatch at {base}: A{a_tuple(a_shape)} B{a_tuple(b_shape)}, "
                f"expected rank {args.rank} on A[0] and B[1]"
            )

    if args.base_model is not None:
        shapes = base_layer_shapes(args.base_model.resolve())
        for base, parts in paired.items():
            key = base_weight_key(parts["A"])
            if key not in shapes:
                raise SystemExit(f"base model has no layer {key}")
            out_features, in_features = shapes[key]
            a_shape = tuple(adapter[parts["A"]].shape)
            b_shape = tuple(adapter[parts["B"]].shape)
            if a_shape[1] != in_features or b_shape[0] != out_features:
                raise SystemExit(
                    f"{key} expects A(r,{in_features}) B({out_features},r) but the adapter has "
                    f"A{a_tuple(a_shape)} B{a_tuple(b_shape)}"
                )
        print(f"checked {len(paired)} adapter modules against {args.base_model}")

    modules = sorted({name.rsplit(".lora_", 1)[0].split(".")[-1] for name in adapter})
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
    (output / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "artifact": "peft_adapter_from_verl_fsdp",
                "shards": [path.name for path in shards],
                "world_size": world_sizes.pop(),
                "tensors": len(adapter),
                "modules": len(paired),
                "rank": args.rank,
                "alpha": args.alpha,
                "shapes": {name: list(shape) for name, shape in sorted(global_shapes.items())},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote PEFT adapter with {len(adapter)} tensors to {output}")
    print("target modules:", config["target_modules"])


def a_tuple(shape) -> tuple[int, ...]:
    return tuple(int(size) for size in shape)


if __name__ == "__main__":
    main()
