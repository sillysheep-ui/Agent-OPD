#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.analysis.gradient import (
    assert_held_out_reference_games,
    gradient_alignment,
)
from omniopd.io import read_jsonl, write_jsonl
from omniopd.loss import weighted_causal_ce
from omniopd.provenance import (
    fingerprint_code_tree,
    fingerprint_path,
    git_revision,
    sha256_file,
)
from omniopd.tokenization import encode_final_assistant_content
from omniopd.prompts import STUDENT_SYSTEM_PROMPT


def _python_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _python_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _python_value(value.tolist())
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = list(read_jsonl(path))
    elif path.suffix == ".parquet":
        import pandas as pd

        rows = [
            {key: _python_value(value) for key, value in row.items()}
            for row in pd.read_parquet(path).to_dict(orient="records")
        ]
    else:
        raise ValueError("M1 data must end in .jsonl or .parquet")
    required = {
        "messages",
        "state_hash",
        "game_id",
        "state_weight",
        "teacher_action",
        "teacher_sample_index",
        "weighting_mode",
        "protocol_version",
        "enable_thinking",
    }
    if not rows or any(required - set(row) for row in rows):
        raise ValueError(f"every M1 row requires {sorted(required)}")
    identities = [
        (str(row["state_hash"]), int(row["teacher_sample_index"])) for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("M1 data contains duplicate state/Teacher-sample identities")
    game_weights: dict[str, float] = {}
    for row in rows:
        messages = row["messages"]
        expected = f"Action: {row.get('teacher_action', '')}"
        roles = [message.get("role") for message in messages]
        expected_roles = [
            "system" if index == 0 else "user" if index % 2 else "assistant"
            for index in range(len(messages))
        ]
        weight = float(row["state_weight"])
        if (
            not messages
            or roles != expected_roles
            or messages[0].get("content") != STUDENT_SYSTEM_PROMPT
            or messages[-1] != {"role": "assistant", "content": expected}
            or row["protocol_version"] != "omniopd-v1"
            or bool(row["enable_thinking"])
            or row["weighting_mode"] != "game_state_mean"
            or not math.isfinite(weight)
            or weight <= 0.0
        ):
            raise ValueError("M1 row violates the canonical action-training contract")
        game = str(row["game_id"])
        game_weights[game] = game_weights.get(game, 0.0) + weight
    if any(
        not math.isclose(weight, 1.0, rel_tol=1e-6, abs_tol=1e-8)
        for weight in game_weights.values()
    ):
        raise ValueError("M1 game_state_mean weights must sum to one within every game")
    return rows


def _gradient_vector(model, parameters, tokenizer, messages, *, max_length: int, device):
    import torch

    encoded = encode_final_assistant_content(
        tokenizer,
        messages,
        max_length=max_length,
        truncation="error",
        enable_thinking=False,
    )
    input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=device)
    target_mask = torch.tensor(
        [encoded.target_token_mask], dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    device_type = torch.device(device).type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    ):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    loss = weighted_causal_ce(
        output.logits,
        input_ids,
        target_mask,
        torch.ones(1, dtype=torch.float32, device=device),
    )
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    pieces = [
        torch.zeros_like(parameter).reshape(-1)
        if gradient is None
        else gradient.reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(pieces).detach().float().cpu()


def _weighted_gradient_mean(rows, gradient_for_row):
    total = None
    total_weight = 0.0
    for row in rows:
        weight = float(row["state_weight"])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("M1 state weights must be finite and positive")
        gradient = gradient_for_row(row)
        total = gradient * weight if total is None else total + gradient * weight
        total_weight += weight
    if total is None or total_weight <= 0:
        raise ValueError("M1 gradient population is empty")
    return total / total_weight


def _parse_group(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("group must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("group must be NAME=PATH")
    return name.strip(), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M1 held-out surrogate-gradient alignment in a fixed LoRA tangent space"
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-audit", required=True)
    parser.add_argument("--group", action="append", type=_parse_group, required=True)
    parser.add_argument(
        "--group-audit", action="append", type=_parse_group, required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-seed", type=int, default=20260831)
    parser.add_argument("--target-modules", default="all-linear")
    args = parser.parse_args()

    base = Path(args.base_model)
    tokenizer_path = Path(args.tokenizer or args.base_model)
    reference_path = Path(args.reference)
    reference_audit_path = Path(args.reference_audit)
    output = Path(args.output_dir)
    group_paths = dict(args.group)
    group_audit_paths = dict(args.group_audit)
    paths = [
        base,
        tokenizer_path,
        reference_path,
        reference_audit_path,
        *group_paths.values(),
        *group_audit_paths.values(),
    ]
    if any(not path.exists() for path in paths):
        raise SystemExit("all model, tokenizer, reference, and group inputs must exist")
    if output.exists():
        raise SystemExit(f"refusing to overwrite M1 output: {output}")
    if args.max_length <= 0 or args.lora_rank <= 0 or args.lora_alpha <= 0:
        raise SystemExit("length, LoRA rank, and LoRA alpha must be positive")
    group_names = [name for name, _ in args.group]
    group_audit_names = [name for name, _ in args.group_audit]
    if (
        len(group_names) != len(set(group_names))
        or len(group_audit_names) != len(set(group_audit_names))
        or set(group_names) != set(group_audit_names)
        or any(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None
        for name in group_names
        )
    ):
        raise SystemExit(
            "M1 group/group-audit names must be identical unique safe components"
        )
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if not current_revision:
        raise SystemExit("M1 analysis requires an immutable Git revision")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if (
        hasattr(model.config, "max_position_embeddings")
        and args.max_length > int(model.config.max_position_embeddings)
    ):
        raise SystemExit("--max-length exceeds the base model's native context window")
    torch.manual_seed(args.lora_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.lora_seed)
    target_modules: str | list[str]
    target_modules = (
        "all-linear"
        if args.target_modules == "all-linear"
        else [value.strip() for value in args.target_modules.split(",") if value.strip()]
    )
    if not target_modules:
        raise SystemExit("--target-modules must contain at least one module name")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="none",
        ),
    ).to(device)
    model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("LoRA setup exposed no trainable parameters")

    reference_rows = _load_rows(reference_path)
    group_rows = {name: _load_rows(path) for name, path in args.group}

    def validate_unsplit_audit(data_path: Path, audit_path: Path, rows: list[dict]) -> dict:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        correction_manifest = audit.get("correction_manifest")
        if (
            audit.get("artifact") != "action_only_training_data_audit"
            or audit.get("protocol_version") != "omniopd-v1"
            or audit.get("code") != current_code
            or audit.get("code_revision") != current_revision
            or audit.get("training_output_sha256") != sha256_file(data_path)
            or audit.get("weighting_mode") != "game_state_mean"
            or "split" not in audit
            or audit.get("split") is not None
            or int(audit.get("validation_rows", -1)) != 0
            or int(audit.get("training_rows", -1)) != len(rows)
            or not isinstance(correction_manifest, dict)
            or not isinstance(correction_manifest.get("sha256"), str)
            or not correction_manifest["sha256"]
        ):
            raise SystemExit(
                f"M1 data {data_path} must be a complete unsplit canonical action table"
            )
        return audit

    reference_audit = validate_unsplit_audit(
        reference_path, reference_audit_path, reference_rows
    )
    group_audits = {
        name: validate_unsplit_audit(
            group_paths[name], group_audit_paths[name], group_rows[name]
        )
        for name in group_names
    }
    reference_games = {str(row["game_id"]) for row in reference_rows}
    correction_games = {
        str(row["game_id"]) for rows in group_rows.values() for row in rows
    }
    assert_held_out_reference_games(correction_games, reference_games)

    def gradient_for_row(row):
        return _gradient_vector(
            model,
            parameters,
            tokenizer,
            row["messages"],
            max_length=args.max_length,
            device=device,
        )

    reference_gradient = _weighted_gradient_mean(reference_rows, gradient_for_row)
    summaries = {}
    individual_outputs: dict[str, list[dict[str, Any]]] = {}
    for name, rows in group_rows.items():
        weighted_sum = None
        total_weight = 0.0
        individual = []
        weighted_cosine = 0.0
        weighted_dot = 0.0
        negative_weight = 0.0
        for row_index, row in enumerate(rows):
            weight = float(row["state_weight"])
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("M1 state weights must be finite and positive")
            gradient = gradient_for_row(row)
            alignment = gradient_alignment(gradient, reference_gradient)
            weighted_sum = (
                gradient * weight
                if weighted_sum is None
                else weighted_sum + gradient * weight
            )
            total_weight += weight
            weighted_cosine += alignment.cosine * weight
            weighted_dot += alignment.dot_product * weight
            negative_weight += float(alignment.dot_product < 0.0) * weight
            individual.append(
                {
                    "group": name,
                    "row_index": row_index,
                    "state_hash": row["state_hash"],
                    "game_id": row["game_id"],
                    "teacher_sample_index": int(row.get("teacher_sample_index", 0)),
                    "state_weight": weight,
                    "dot_product": alignment.dot_product,
                    "cosine": alignment.cosine,
                    "correction_gradient_norm": alignment.correction_norm,
                }
            )
        if weighted_sum is None or total_weight <= 0:
            raise ValueError(f"M1 group {name!r} is empty")
        group_gradient = weighted_sum / total_weight
        group_alignment = gradient_alignment(group_gradient, reference_gradient)
        summaries[name] = {
            "rows": len(rows),
            "games": len({str(row["game_id"]) for row in rows}),
            "total_objective_weight": total_weight,
            "weighted_mean_individual_dot": weighted_dot / total_weight,
            "weighted_mean_individual_cosine": weighted_cosine / total_weight,
            "negative_dot_weight_fraction": negative_weight / total_weight,
            "dataset_dot_product": group_alignment.dot_product,
            "dataset_cosine": group_alignment.cosine,
            "dataset_gradient_norm": group_alignment.correction_norm,
            "reference_gradient_norm": group_alignment.reference_norm,
        }
        individual_outputs[name] = individual

    output.mkdir(parents=True)
    torch.save(reference_gradient, output / "reference_gradient.pt")
    individual_hashes = {}
    for name, rows in individual_outputs.items():
        path = output / f"individual_{name}.jsonl"
        write_jsonl(path, rows)
        individual_hashes[name] = sha256_file(path)
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "m1_surrogate_gradient_alignment",
        "interpretation": "held_out_teacher_ce_surrogate_not_task_success_gradient",
        "code": current_code,
        "code_revision": current_revision,
        "base_model": fingerprint_path(base),
        "tokenizer": fingerprint_path(tokenizer_path),
        "reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "audit_path": str(reference_audit_path),
            "audit_sha256": sha256_file(reference_audit_path),
            "correction_manifest": reference_audit["correction_manifest"],
            "rows": len(reference_rows),
            "games": len(reference_games),
        },
        "groups": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "audit_path": str(group_audit_paths[name]),
                "audit_sha256": sha256_file(group_audit_paths[name]),
                "correction_manifest": group_audits[name]["correction_manifest"],
            }
            for name, path in args.group
        },
        "token_contract": "assistant_action_content_tokens_v1",
        "enable_thinking": False,
        "model_master_dtype": "fp32",
        "cuda_forward_dtype": "bf16",
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": 0.0,
            "seed": args.lora_seed,
            "target_modules": target_modules,
        },
        "summaries": summaries,
        "outputs": {
            "reference_gradient_sha256": sha256_file(
                output / "reference_gradient.pt"
            ),
            "individual_sha256": individual_hashes,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
