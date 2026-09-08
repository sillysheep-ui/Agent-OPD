#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniopd.io import read_jsonl, write_jsonl
from omniopd.loss import mean_target_log_probability
from omniopd.evaluation import (
    validate_artifact_fingerprint,
    validate_training_completion_manifest,
)
from omniopd.provenance import (
    fingerprint_code_tree,
    fingerprint_path,
    git_revision,
    sha256_file,
)
from omniopd.tokenization import encode_final_assistant_content
from omniopd.validation import validate_lora_checkpoint_directory


def _training_comparison_contract(manifest: dict) -> dict:
    """Extract the preregistered fields that must match across M3 checkpoints."""

    training = manifest.get("training_contract")
    hyperparameters = manifest.get("hyperparameters")
    if not isinstance(training, dict) or not isinstance(hyperparameters, dict):
        raise ValueError("training contract/hyperparameters are missing")
    training_keys = [
        "global_train_batch_size",
        "local_train_batch_size",
        "micro_batch_size_per_gpu",
        "microbatches_per_optimizer_step",
        "total_optimizer_steps",
        "seed",
        "model_load_dtype",
        "forward_and_fsdp_param_dtype",
        "cross_entropy_dtype",
        "gradient_reduce_dtype",
        "attention_implementation",
        "distributed_strategy",
        "state_weight_normalization",
        "sampler_padding_weight",
        "encoded_sequence_overflow_policy",
    ]
    hyperparameter_keys = [
        "learning_rate",
        "max_length",
        "lora_rank",
        "lora_alpha",
        "target_modules",
    ]
    missing = [key for key in training_keys if key not in training]
    missing += [
        f"hyperparameters.{key}"
        for key in hyperparameter_keys
        if key not in hyperparameters
    ]
    if missing:
        raise ValueError(f"training comparison contract lacks fields: {missing}")
    return {
        "training_contract": {key: training[key] for key in training_keys},
        "hyperparameters": {
            key: hyperparameters[key] for key in hyperparameter_keys
        },
    }


def _python_value(value):
    if isinstance(value, dict):
        return {key: _python_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _python_value(value.tolist())
    return value


def _load_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return list(read_jsonl(path))
    if path.suffix == ".parquet":
        import pandas as pd

        return [
            {key: _python_value(value) for key, value in row.items()}
            for row in pd.read_parquet(path).to_dict(orient="records")
        ]
    raise SystemExit("--data must end in .jsonl or .parquet")


def _resolve_training_tokenizer_path(
    base_model_path: Path, tokenizer_argument: str | None
) -> Path:
    """Return the tokenizer path proven to be the training tokenizer.

    The audited trainer loads its tokenizer from ``model.partial_pretrain``.
    M3 therefore cannot accept an independently supplied tokenizer, even when
    all six score cells happen to share that alternative tokenizer.
    """

    tokenizer_path = (
        Path(tokenizer_argument) if tokenizer_argument is not None else base_model_path
    )
    if tokenizer_path.resolve() != base_model_path.resolve():
        raise ValueError(
            "M3 scoring tokenizer must be the exact base-model directory used by "
            "training; an independent tokenizer would change the action-token estimand"
        )
    return tokenizer_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score training-identical final action log probability for M3"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument(
        "--training-manifest",
        help="canonical launch_manifest.json required with --adapter",
    )
    parser.add_argument(
        "--training-completion-manifest",
        help="canonical completion_manifest.json required with --adapter",
    )
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--checkpoint-group", choices=["BASE", "A1", "A3"], required=True
    )
    parser.add_argument("--panel-group", choices=["A1", "A3"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--model-dtype", choices=["fp32", "bf16"], default="bf16")
    args = parser.parse_args()

    output = Path(args.output)
    manifest_path = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite an existing score or manifest file")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    base_model_path = Path(args.base_model)
    adapter_path = Path(args.adapter) if args.adapter else None
    training_manifest_path = (
        Path(args.training_manifest) if args.training_manifest else None
    )
    training_completion_path = (
        Path(args.training_completion_manifest)
        if args.training_completion_manifest
        else None
    )
    if len(
        {
            adapter_path is None,
            training_manifest_path is None,
            training_completion_path is None,
        }
    ) != 1:
        raise SystemExit(
            "--adapter, --training-manifest, and --training-completion-manifest "
            "must be supplied together"
        )
    if (adapter_path is None) != (args.checkpoint_group == "BASE"):
        raise SystemExit(
            "BASE scoring must omit an adapter; A1/A3 scoring must load one"
        )
    try:
        tokenizer_path = _resolve_training_tokenizer_path(
            base_model_path, args.tokenizer
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    required_paths = [base_model_path, tokenizer_path]
    if adapter_path is not None:
        required_paths.append(adapter_path)
        required_paths.append(training_manifest_path)
        required_paths.append(training_completion_path)
        required_paths.append(training_manifest_path.parent / "resolved_config.yaml")
        required_paths.append(training_manifest_path.parent / "train.log")
    if any(not path.exists() for path in required_paths):
        raise SystemExit("base model, optional adapter, and tokenizer must be local paths")
    current_code = fingerprint_code_tree(ROOT)
    current_revision = git_revision(ROOT)
    if not current_revision:
        raise SystemExit("M3 scoring requires an immutable Git revision")
    base_fingerprint = fingerprint_path(base_model_path)
    adapter_fingerprint = fingerprint_path(adapter_path) if adapter_path else None
    training_identity = None
    if training_manifest_path is not None:
        training_manifest = json.loads(
            training_manifest_path.read_text(encoding="utf-8")
        )
        training_completion = json.loads(
            training_completion_path.read_text(encoding="utf-8")
        )
        try:
            completion_identity = validate_training_completion_manifest(
                training_completion,
                training_manifest,
                launch_manifest_sha256=sha256_file(training_manifest_path),
                completion_manifest_sha256=sha256_file(training_completion_path),
                checkpoint_fingerprint=adapter_fingerprint,
                current_checkpoint_format=validate_lora_checkpoint_directory(
                    adapter_path,
                    expected_rank=int(training_manifest["hyperparameters"]["lora_rank"]),
                    expected_alpha=float(training_manifest["hyperparameters"]["lora_alpha"]),
                    expected_target_modules_policy=str(
                        training_manifest["hyperparameters"]["target_modules"]
                    ),
                ),
                resolved_config_fingerprint=fingerprint_path(
                    training_manifest_path.parent / "resolved_config.yaml"
                ),
                training_log_fingerprint=fingerprint_path(
                    training_manifest_path.parent / "train.log"
                ),
            )
            total_steps = int(completion_identity["total_optimizer_steps"])
            training_seed = int(completion_identity["training_seed"])
            comparison_contract = _training_comparison_contract(training_manifest)
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"M3 training manifest is incomplete: {error}") from error
        try:
            validate_artifact_fingerprint(
                training_manifest.get("inputs", {}).get("model_path", {}),
                base_fingerprint,
                role="M3 scoring base/training base",
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        expected_adapter = training_manifest_path.parent / f"global_step_{total_steps}"
        if (
            training_manifest.get("artifact") != "omniopd_training_launch"
            or training_manifest.get("protocol_version") != "omniopd-v1"
            or training_manifest.get("code") != current_code
            or training_manifest.get("code_revision") != current_revision
            or adapter_path.resolve() != expected_adapter.resolve()
            or not isinstance(training_manifest.get("experiment"), str)
            or not training_manifest["experiment"].strip()
        ):
            raise SystemExit(
                "M3 adapter must be the final checkpoint of a canonical training run "
                "that used this exact base model and code revision"
            )
        training_identity = {
            "path": str(training_manifest_path),
            "sha256": sha256_file(training_manifest_path),
            "completion_path": str(training_completion_path),
            "completion_sha256": sha256_file(training_completion_path),
            "experiment": completion_identity["experiment"],
            "training_seed": training_seed,
            "total_optimizer_steps": total_steps,
            "comparison_contract": comparison_contract,
            "training_inputs": completion_identity["inputs"],
            "arm_contract": completion_identity["arm_contract"],
            "final_checkpoint": completion_identity["final_checkpoint"],
            "final_checkpoint_verified": True,
        }

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise SystemExit("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_dtype == "bf16" and device.startswith("cuda"):
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("--model-dtype bf16 requires bf16-capable CUDA")
    model_dtype = torch.float32 if args.model_dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model, adapter_path, autocast_adapter_dtype=False
        )
    model = model.to(device=device, dtype=model_dtype)
    model.eval()

    rows = _load_rows(Path(args.data))
    required = {
        "messages",
        "score_id",
        "source_id",
        "group",
        "panel_group",
        "panel_experiment",
        "target_kind",
        "source_state_hash",
        "target_state_hash",
        "state_hash",
        "game_id",
        "turn_index",
        "teacher_sample_index",
        "teacher_action",
        "protocol_version",
        "enable_thinking",
    }
    if not rows or any(required - set(row) for row in rows):
        raise SystemExit(f"every data row must contain {sorted(required)}")
    score_ids = [str(row["score_id"]) for row in rows]
    if len(score_ids) != len(set(score_ids)):
        raise SystemExit("M3 score rows must have unique score IDs")
    for row in rows:
        if (
            row["protocol_version"] != "omniopd-v1"
            or bool(row["enable_thinking"])
            or row["group"] not in {"A1", "A3"}
            or row["group"] != args.panel_group
            or row["panel_group"] != args.panel_group
            or not isinstance(row["panel_experiment"], str)
            or not row["panel_experiment"].strip()
            or row["target_kind"] not in {"self", "neighbor"}
            or row["state_hash"] != row["target_state_hash"]
            or row["messages"][-1]
            != {"role": "assistant", "content": f"Action: {row['teacher_action']}"}
        ):
            raise SystemExit("M3 score row violates its frozen identity/action contract")
    if hasattr(model.config, "max_position_embeddings"):
        native_context = int(model.config.max_position_embeddings)
    else:
        native_context = None
    results = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        encoded = [
            encode_final_assistant_content(
                tokenizer,
                row["messages"],
                truncation="error",
                enable_thinking=False,
            )
            for row in batch_rows
        ]
        width = max(len(item.input_ids) for item in encoded)
        if native_context is not None and width > native_context:
            raise SystemExit(
                f"M3 encoded sequence length {width} exceeds native context {native_context}"
            )
        input_ids = torch.full(
            (len(encoded), width), tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        target_mask = torch.zeros(
            (len(encoded), width), dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            (len(encoded), width), dtype=torch.long, device=device
        )
        for index, item in enumerate(encoded):
            length = len(item.input_ids)
            input_ids[index, :length] = torch.tensor(item.input_ids, device=device)
            attention_mask[index, :length] = 1
            target_mask[index, :length] = torch.tensor(
                item.target_token_mask, device=device
            )
        position_ids = torch.arange(width, device=device).unsqueeze(0) * attention_mask
        scores = mean_target_log_probability(
            model,
            input_ids,
            target_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).cpu().tolist()
        for row, item, score in zip(batch_rows, encoded, scores):
            results.append(
                {
                    "protocol_version": "omniopd-v1",
                    "base_model": str(base_model_path),
                    "adapter": str(adapter_path) if adapter_path else None,
                    "score_id": row["score_id"],
                    "source_id": row["source_id"],
                    "group": row["group"],
                    "panel_group": row["panel_group"],
                    "panel_experiment": row["panel_experiment"],
                    "checkpoint_group": args.checkpoint_group,
                    "checkpoint_experiment": (
                        training_identity["experiment"] if training_identity else None
                    ),
                    "target_kind": row["target_kind"],
                    "source_state_hash": row["source_state_hash"],
                    "target_state_hash": row["target_state_hash"],
                    "state_hash": row["state_hash"],
                    "game_id": row["game_id"],
                    "turn_index": int(row["turn_index"]),
                    "teacher_sample_index": int(row.get("teacher_sample_index", 0)),
                    "teacher_action": row["teacher_action"],
                    "mean_target_log_probability": float(score),
                    "target_tokens": item.target_length,
                }
            )
    write_jsonl(output, results)
    manifest = {
        "protocol_version": "omniopd-v1",
        "artifact": "action_imitation_log_probability",
        "code": current_code,
        "code_revision": current_revision,
        "base_model": base_fingerprint,
        "adapter": adapter_fingerprint,
        "training_manifest": training_identity,
        "checkpoint_group": args.checkpoint_group,
        "checkpoint_experiment": (
            training_identity["experiment"] if training_identity else None
        ),
        "panel_group": args.panel_group,
        "panel_experiment": rows[0]["panel_experiment"],
        "tokenizer": fingerprint_path(tokenizer_path),
        "training_tokenizer_verified": True,
        "training_tokenizer_source": "base_model_directory_model.partial_pretrain",
        "model_dtype": args.model_dtype,
        "device": device,
        "rows": len(results),
        "data_sha256": sha256_file(args.data),
        "scores_sha256": sha256_file(output),
        "token_contract": "assistant_action_content_tokens_v1",
        "enable_thinking": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
