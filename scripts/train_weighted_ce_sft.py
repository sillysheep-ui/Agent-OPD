#!/usr/bin/env python3
"""Main-line step 5: weighted-CE SFT on collected Teacher-action rows.

Runs the objective of AGENT_OPD.md §2.4 through the repository's own
`weighted_causal_ce_components`, and verifies the three things a training step
must show before any evaluation is worth running:

  A. loss wiring  -- the batch loss equals a hand-computed weighted CE
  B. gradients    -- grad norm is non-zero at the first step
  C. update size  -- ||dW||_F / ||W||_F over the LoRA target modules lands in
                     the band the main line requires (1e-3 .. 1e-2)

Only action-content tokens are supervised (the mask comes from step 4), and
`state_weight` carries the game/state/sample normalisation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.loss import weighted_causal_ce  # noqa: E402

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def manual_weighted_ce(logits, input_ids, mask, weights) -> float:
    """Independent recomputation of the batch objective (verification A)."""

    token_loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]).float(),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(input_ids.shape[0], -1)
    aligned = mask[:, 1:].float()
    per_state = (token_loss * aligned).sum(dim=1) / aligned.sum(dim=1)
    return float((per_state * weights).sum() / weights.sum())


def delta_ratio(adapter_state: dict, base_dir: Path, rank: int, alpha: int) -> float:
    """||dW||_F / ||W||_F over the adapter's target modules (verification C)."""

    from safetensors import safe_open

    scale = alpha / rank
    grouped: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, tensor in adapter_state.items():
        if ".lora_A" in key:
            grouped[key.split(".lora_A")[0]]["A"] = tensor.float()
        elif ".lora_B" in key:
            grouped[key.split(".lora_B")[0]]["B"] = tensor.float()
    base_files = sorted(base_dir.glob("*.safetensors"))
    delta_sq = base_sq = 0.0
    for module, parts in grouped.items():
        if "A" not in parts or "B" not in parts:
            continue
        a, b = parts["A"], parts["B"]
        delta_sq += scale * scale * float((a @ a.t() * (b.t() @ b).t()).sum())
        # base weight key: strip the peft prefix and the adapter suffix
        name = module.replace("base_model.model.", "").replace(".base_layer", "")
        key = f"{name}.weight"
        for path in base_files:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if key in handle.keys():
                    base_sq += float(handle.get_tensor(key).float().pow(2).sum())
                    break
    return math.sqrt(delta_sq) / math.sqrt(base_sq)


def collate(rows: list[dict], device: torch.device, pad_id: int):
    width = max(len(row["input_ids"]) for row in rows)
    input_ids, masks, weights = [], [], []
    for row in rows:
        ids = row["input_ids"]
        mask = row["target_token_mask"]
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        masks.append(mask + [0] * pad)
        weights.append(float(row["state_weight"]))
    return (
        torch.tensor(input_ids, device=device),
        torch.tensor(masks, device=device),
        torch.tensor(weights, device=device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("no training rows")
    tokenizer = AutoTokenizer.from_pretrained(str(args.student_model), local_files_only=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        str(args.student_model), dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=list(TARGET_MODULES),
        ),
    )
    model.config.use_cache = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    batches = [rows[i : i + args.micro_batch] for i in range(0, len(rows), args.micro_batch)]
    weights_sum = sum(float(row["state_weight"]) for row in rows)
    per_game = defaultdict(float)
    for row in rows:
        per_game[row["game_id"]] += float(row["state_weight"])
    print(f"[rows] {len(rows)} rows / {len(per_game)} games, Σstate_weight={weights_sum:.6f}, "
          f"per-game " + ", ".join(f"{g.split('/')[-1][:18]}={w:.6f}" for g, w in sorted(per_game.items())))

    # verification A on a tiny synthetic batch: the wiring check must not double
    # the fp32 logits of a 4.6k-token row inside the real training loop.
    probe = [
        {
            "input_ids": rows[0]["input_ids"][-64:],
            "target_token_mask": rows[0]["target_token_mask"][-64:],
            "state_weight": float(rows[0]["state_weight"]),
        },
        {
            "input_ids": rows[-1]["input_ids"][-64:],
            "target_token_mask": rows[-1]["target_token_mask"][-64:],
            "state_weight": float(rows[-1]["state_weight"]),
        },
    ]
    probe_ids, probe_mask, probe_weights = collate(probe, device, tokenizer.pad_token_id or 0)
    with torch.no_grad():
        probe_logits = model(input_ids=probe_ids).logits
        probe_loss = weighted_causal_ce(probe_logits, probe_ids, probe_mask, probe_weights)
        probe_reference = manual_weighted_ce(probe_logits, probe_ids, probe_mask, probe_weights)
    probe_delta = abs(float(probe_loss) - probe_reference)
    print(f"[A] loss wiring : probe loss={float(probe_loss):.8f} manual={probe_reference:.8f} "
          f"|diff|={probe_delta:.3e} -> {'OK' if probe_delta < 1e-4 else 'FAIL'}")
    if probe_delta >= 1e-4:
        raise SystemExit("loss wiring check failed")
    del probe_logits
    torch.cuda.empty_cache()

    history = []
    grad_norm_first = None
    model.train()
    for epoch in range(args.epochs):
        for index, batch_rows in enumerate(batches):
            input_ids, mask, weights = collate(batch_rows, device, tokenizer.pad_token_id or 0)
            out = model(input_ids=input_ids)
            loss = weighted_causal_ce(out.logits, input_ids, mask, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == 0 and index == 0:
                grad_norm_first = float(
                    torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
                )
                print(f"[B] gradients   : grad_norm(first step)={grad_norm_first:.4e} -> "
                      f"{'OK' if grad_norm_first > 0 else 'FAIL'}")
                if not grad_norm_first > 0:
                    raise SystemExit("no gradient reached the LoRA parameters")
            optimizer.step()
            history.append(float(loss))
            print(f"    epoch {epoch + 1}/{args.epochs} batch {index + 1}/{len(batches)} loss={float(loss):.6f}")

    first, last = history[0], history[-1]
    print(f"[loss] first={first:.6f} last={last:.6f} min={min(history):.6f} "
          f"-> {'下降' if last < first else '未下降'}")

    model.save_pretrained(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "adapter"))

    # verification C: relative weight change of the trained adapter
    from safetensors.torch import load_file

    adapter_state = load_file(str(args.output_dir / "adapter" / "adapter_model.safetensors"))
    ratio = delta_ratio(adapter_state, args.student_model, args.lora_rank, args.lora_alpha)
    band = 1e-3 <= ratio <= 1e-2
    print(f"[C] update size : ||dW||_F/||W||_F={ratio:.3e} (band 1e-3..1e-2) -> "
          f"{'OK' if band else 'OUT OF BAND'}")
    (args.output_dir / "training_report.json").write_text(json.dumps({
        "rows": len(rows), "games": len(per_game), "epochs": args.epochs, "lr": args.lr,
        "loss_first": first, "loss_last": last, "loss_min": min(history),
        "grad_norm_first_step": grad_norm_first, "delta_ratio": ratio,
        "state_weight_sum": weights_sum, "per_game_weight": dict(per_game),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"    adapter saved to {args.output_dir / 'adapter'}")


if __name__ == "__main__":
    main()
