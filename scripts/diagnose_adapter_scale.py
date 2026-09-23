#!/usr/bin/env python3
"""Measure how far a trained LoRA adapter actually moved the model weights.

"Loss looked reasonable" says nothing about how much of the model changed. This
script reports, per target module and in aggregate, the Frobenius norm of the
adapter's weight update relative to the base layer it modifies:

    delta_W = (alpha / r) * B @ A        compared against ||W||_F

The update norm is computed without materializing delta_W, using
||B A||_F^2 = <A A^T, B^T B>_F, so it costs almost nothing even for 4096-wide
layers. A ratio far below 1e-3 means the run could not have changed behaviour,
and the number of optimizer steps needed to reach a useful ratio follows from
the measured growth per step.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

BASE_KEY_PREFIXES = ("base_model.model.", "base_model.")


def base_weight_key(adapter_key: str) -> str:
    """Map a PEFT adapter key onto the base layer it modifies."""

    for prefix in BASE_KEY_PREFIXES:
        if adapter_key.startswith(prefix):
            return adapter_key[len(prefix):].rsplit(".lora_", 1)[0] + ".weight"
    raise ValueError(f"adapter key has no known base prefix: {adapter_key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    adapter_dir = args.adapter.resolve()
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    rank = int(config["r"])
    scale = float(config["lora_alpha"]) / rank
    weights = load_file(str(adapter_dir / "adapter_model.safetensors"))

    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, tensor in weights.items():
        base, tail = key.rsplit(".lora_", 1)
        pairs[base][tail.split(".")[0]] = tensor

    base_files = sorted(args.base_model.resolve().glob("*.safetensors"))
    if not base_files:
        raise SystemExit(f"no base safetensors under {args.base_model}")
    base_shapes = {}
    for path in base_files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                base_shapes[key] = path

    per_module = []
    delta_sq_total = 0.0
    base_sq_total = 0.0
    b_norm_max = 0.0
    for base, parts in sorted(pairs.items()):
        a = parts["A"].float()
        b = parts["B"].float()
        # ||B A||_F^2 = <A A^T, B^T B>_F
        aa = a @ a.t()
        bt_b = b.t() @ b
        delta_sq = float((scale * scale) * (aa * bt_b.t()).sum())
        key = base_weight_key(f"{base}.lora_A.weight")
        if key not in base_shapes:
            raise SystemExit(f"base model has no layer {key}")
        with safe_open(str(base_shapes[key]), framework="pt", device="cpu") as handle:
            w = handle.get_tensor(key).float()
        base_sq = float(w.pow(2).sum())
        per_module.append(
            {
                "module": key,
                "delta_fro": delta_sq**0.5,
                "base_fro": base_sq**0.5,
                "relative": (delta_sq**0.5) / (base_sq**0.5),
                "b_fro": float(b.norm()),
                "a_fro": float(a.norm()),
            }
        )
        delta_sq_total += delta_sq
        base_sq_total += base_sq
        b_norm_max = max(b_norm_max, float(b.norm()))

    relative_total = (delta_sq_total**0.5) / (base_sq_total**0.5)
    per_module.sort(key=lambda row: row["relative"], reverse=True)
    payload = {
        "artifact": "adapter_scale_diagnostic",
        "adapter": str(adapter_dir),
        "rank": rank,
        "lora_alpha": config["lora_alpha"],
        "scale": scale,
        "modules": len(per_module),
        "delta_fro_total": delta_sq_total**0.5,
        "base_fro_total": base_sq_total**0.5,
        "relative_change_total": relative_total,
        "largest_b_fro": b_norm_max,
        "relative_change_by_module": {
            row["module"].split("model.layers.")[-1].split(".")[0] + ":" + row["module"].rsplit(".", 2)[-2]:
            row["relative"]
            for row in per_module[: args.top]
        },
        "per_module": per_module,
    }
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k != "per_module"}, indent=2
        )
    )
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
