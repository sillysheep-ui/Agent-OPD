# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audited veRL FSDP trainer for Agent OmniOPD.

This file deliberately lives in the repository instead of patching an installed
``verl.trainer.fsdp_sft_trainer`` in place.  Its loss contract is:

* ``target_token_mask[:, t]`` supervises ``input_ids[:, t]`` and is therefore
  aligned with causal labels using ``[:, 1:]``;
* under uniform row sampling, ``state_weight`` uses the fixed expected sampled
  mass as its normalizer, yielding an unbiased estimate of the full weighted
  dataset objective even when the optimizer batch contains one real row;
* zero-weight sampler padding equalizes collective call counts without changing
  either the training or validation objective.

Sequence parallel/remove-padding execution is intentionally rejected below.  It
needs a separate packed-sequence boundary audit before it can share this loss
implementation safely.
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import math
import random
from collections.abc import Iterator, Mapping
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch import optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking

from omniopd.loss import weighted_causal_ce_components
from omniopd.sampler import ZeroPaddedDistributedSampler
from omniopd.torch_dataset import FinalTurnActionDataset
from omniopd.verl_adapter import uniform_sampling_normalizer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))

def _seed_everything(seed: int) -> None:
    """Seed every RNG used directly by this trainer."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _BatchAlignedSampler(Sampler):
    """Pad each rank to a whole local batch with zero-weight examples.

    ``ZeroPaddedDistributedSampler`` first makes every rank equally long.  A
    second, local padding step prevents the final optimizer batch from changing
    tensor batch size.  All additional indices carry multiplier 0, so no real
    example is duplicated in the objective.
    """

    def __init__(self, base: ZeroPaddedDistributedSampler, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(base) <= 0:
            raise ValueError("cannot sample an empty dataset")
        self.base = base
        self.batch_size = int(batch_size)
        self.num_samples = math.ceil(len(base) / self.batch_size) * self.batch_size

    def __iter__(self) -> Iterator[tuple[int, float]]:
        entries = list(iter(self.base))
        if not entries:
            return iter(())
        anchor_index = int(entries[0][0])
        entries.extend(
            (anchor_index, 0.0) for _ in range(self.num_samples - len(entries))
        )
        return iter(entries)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)


def _tensor_batch_size(batch: Mapping) -> int:
    input_ids = batch.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim < 1:
        raise ValueError("batch must contain tensor input_ids with a batch dimension")
    return int(input_ids.shape[0])


def _iter_microbatches(batch: Mapping, micro_batch_size: int) -> Iterator[dict]:
    """Slice tensor fields only; string provenance fields stay in the dataset."""

    batch_size = _tensor_batch_size(batch)
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        yield {
            key: value[start:end]
            for key, value in batch.items()
            if torch.is_tensor(value)
        }


class FSDPSFTTrainer:
    def __init__(self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh, tokenizer, train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.tokenizer = tokenizer
        self.device_name = get_device_name()
        self.seed = int(getattr(self.config.trainer, "seed", 0))
        self.config.ulysses_sequence_parallel_size = int(
            getattr(self.config, "ulysses_sequence_parallel_size", 1)
        )
        self.use_remove_padding = bool(getattr(self.config, "use_remove_padding", False))
        if self.config.ulysses_sequence_parallel_size != 1 or self.use_remove_padding:
            raise NotImplementedError(
                "the audited weighted trainer currently requires "
                "ulysses_sequence_parallel_size=1 and use_remove_padding=false"
            )
        if self.config.data.get("chat_template", None) is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")
        if not isinstance(train_dataset, FinalTurnActionDataset) or not isinstance(
            val_dataset, FinalTurnActionDataset
        ):
            raise TypeError(
                "the audited trainer requires omniopd.torch_dataset.FinalTurnActionDataset"
            )
        overlap = train_dataset.game_ids & val_dataset.game_ids
        if overlap:
            raise ValueError(
                "train/validation game leakage detected: "
                f"{sorted(overlap)[:5]}"
            )

        # normalize dp size
        self._normalize_config_bsz()
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size()
        global_batch_size = int(self.config.data.train_batch_size)
        micro_batch_size = int(self.config.data.micro_batch_size_per_gpu)
        if global_batch_size <= 0 or micro_batch_size <= 0:
            raise ValueError("global and micro batch sizes must be positive")
        if global_batch_size % dp_size != 0:
            raise ValueError(
                f"global batch size {global_batch_size} is not divisible by "
                f"data-parallel size {dp_size}"
            )
        local_batch_size = global_batch_size // dp_size
        if local_batch_size % micro_batch_size != 0:
            raise ValueError(
                f"local batch size {local_batch_size} is not divisible by "
                f"micro batch size {micro_batch_size}"
            )
        self.global_train_batch_size = global_batch_size
        self.local_train_batch_size = local_batch_size
        self.microbatches_per_step = local_batch_size // micro_batch_size
        self.config.data.train_batch_size = local_batch_size
        if self.device_mesh.get_rank() == 0:
            print(
                "Batch contract: "
                f"global={global_batch_size}, local={local_batch_size}, "
                f"micro={micro_batch_size}, "
                f"microbatches/step={self.microbatches_per_step}"
            )

    def _build_dataloader(self, train_dataset, val_dataset):
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.train_objective_weight_sum = float(train_dataset.objective_weight_sum)
        self.train_objective_population_size = int(
            train_dataset.objective_population_size
        )
        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        train_base_sampler = ZeroPaddedDistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            seed=self.seed,
        )
        self.train_sampler = _BatchAlignedSampler(
            train_base_sampler, batch_size=int(config.data.train_batch_size)
        )
        num_workers = int(config.data.get("num_workers", 8))
        train_generator = torch.Generator()
        train_generator.manual_seed(self.seed + rank)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            generator=train_generator,
        )

        val_base_sampler = ZeroPaddedDistributedSampler(
            self.val_dataset,
            shuffle=False,
            num_replicas=world_size,
            rank=rank,
            seed=self.seed,
        )
        self.val_sampler = _BatchAlignedSampler(
            val_base_sampler,
            batch_size=int(config.data.micro_batch_size_per_gpu),
        )
        val_generator = torch.Generator()
        val_generator.manual_seed(self.seed + world_size + rank)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            generator=val_generator,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        model_dtype_name = str(
            self.config.model.fsdp_config.get("model_dtype", "fp32")
        )
        if model_dtype_name != "fp32":
            raise ValueError(
                "model.fsdp_config.model_dtype must be fp32 so AdamW/FSDP retain "
                "full-precision master parameters; got "
                f"{model_dtype_name!r}"
            )
        torch_dtype = PrecisionType.to_dtype(model_dtype_name)
        self.model_load_dtype = model_dtype_name
        self.training_dtype = "bf16"
        if self.device_name == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("the audited bf16 training path requires bf16-capable CUDA")
        if self.device_mesh.get_rank() == 0:
            print(
                f"Precision contract: model-load={self.model_load_dtype}, "
                "forward/FSDP-param=bf16, CE/reduction=fp32"
            )
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            native_context = int(self.model_config.max_position_embeddings)
            if int(self.config.data.max_length) > native_context:
                raise ValueError(
                    f"data.max_length={self.config.data.max_length} exceeds the model's "
                    f"native max_position_embeddings={native_context}; configure an "
                    "audited RoPE-scaling model instead of mutating the architecture"
                )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))
                self.model = self.model.to(torch_dtype)  # unify LoRA params dtype (PEFT defaults fp32)

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy != "fsdp":
            raise NotImplementedError(
                "the audited OmniOPD trainer currently supports FSDP1 only"
            )
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                # Required for PEFT's mixture of frozen base parameters and
                # trainable LoRA parameters within FSDP wrapping boundaries.
                use_orig_params=True,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        configured_total_steps = self.config.trainer.get(
            "total_training_steps", None
        )
        if configured_total_steps is None:
            raise ValueError(
                "trainer.total_training_steps is required: compared runs must use "
                "the same explicit optimizer-step budget"
            )
        self.total_steps = int(configured_total_steps)
        if self.total_steps <= 0:
            raise ValueError("trainer.total_training_steps must be a positive integer")
        self.total_training_steps = self.total_steps

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/dataset pass {self.steps_per_epoch}; "
                f"fixed optimizer-step budget {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _batch_tensors_to_device(self, batch: Mapping) -> dict[str, torch.Tensor]:
        required = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "target_token_mask",
            "state_weight",
            "sampling_multiplier",
        }
        missing = required - set(batch)
        if missing:
            raise ValueError(f"training batch is missing fields: {sorted(missing)}")
        tensors = {
            key: value.to(self.device_name, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        input_ids = tensors["input_ids"]
        attention_mask = tensors["attention_mask"]
        position_ids = tensors["position_ids"]
        target_mask = tensors["target_token_mask"]
        state_weights = tensors["state_weight"].float()
        sampling_multiplier = tensors["sampling_multiplier"].float()
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,T]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must have the same shape as input_ids")
        if target_mask.shape != input_ids.shape:
            raise ValueError("target_token_mask must have the same shape as input_ids")
        if state_weights.ndim != 1 or state_weights.shape[0] != input_ids.shape[0]:
            raise ValueError("state_weight must have shape [B]")
        if (
            sampling_multiplier.ndim != 1
            or sampling_multiplier.shape[0] != input_ids.shape[0]
        ):
            raise ValueError("sampling_multiplier must have shape [B]")
        if not bool(torch.isfinite(state_weights).all()):
            raise ValueError("state_weight contains a non-finite value")
        if bool((state_weights < 0).any()):
            raise ValueError("state_weight must be non-negative")
        target_lengths = target_mask[:, 1:].sum(dim=1)
        if bool((target_lengths <= 0).any()):
            raise ValueError("every sample must retain at least one target token")
        tensors["state_weight"] = state_weights
        return tensors

    def _global_optimizer_batch_normalizer(self, batch: Mapping) -> torch.Tensor:
        """Use expected sampled mass, not a biased random self-normalizer."""

        multipliers = batch.get("sampling_multiplier")
        if not torch.is_tensor(multipliers):
            raise ValueError("batch must contain tensor sampling_multiplier")
        multipliers = multipliers.to(
            self.device_name, dtype=torch.float32, non_blocking=True
        )
        if multipliers.shape != (_tensor_batch_size(batch),):
            raise ValueError("sampling_multiplier must have shape [B]")
        return uniform_sampling_normalizer(
            multipliers,
            dataset_total_weight=self.train_objective_weight_sum,
            dataset_size=self.train_objective_population_size,
            distributed=True,
        )

    def _weighted_forward_components(
        self, batch: Mapping
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = self._batch_tensors_to_device(batch)
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            output = self.fsdp_model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                position_ids=tensors["position_ids"],
                use_cache=False,
            )
            return weighted_causal_ce_components(
                output.logits,
                tensors["input_ids"],
                tensors["target_token_mask"],
                tensors["state_weight"],
            )

    def training_step(self, batch: Mapping):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        optimizer_batch_normalizer = self._global_optimizer_batch_normalizer(batch)
        dp_size = self.device_mesh.size()
        local_scaled_loss = torch.zeros(
            (), device=self.device_name, dtype=torch.float32
        )
        for micro_batch in _iter_microbatches(
            batch, int(self.config.data.micro_batch_size_per_gpu)
        ):
            numerator, _ = self._weighted_forward_components(micro_batch)
            # FSDP averages gradients across DP ranks. The dp_size factor turns
            # that average into the gradient of the global weighted numerator.
            contribution = numerator / optimizer_batch_normalizer * dp_size
            contribution.backward()
            local_scaled_loss += contribution.detach().float()

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # A silent skip changes the effective optimizer-step budget. Fail fast
        # so a compared arm cannot continue with fewer updates than recorded.
        grad_norm_value = (
            float(grad_norm.detach().float().item())
            if torch.is_tensor(grad_norm)
            else float(grad_norm)
        )
        if not math.isfinite(grad_norm_value):
            self.optimizer.zero_grad()
            raise FloatingPointError(f"grad_norm is not finite: {grad_norm_value}")
        self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        lr = self.lr_scheduler.get_last_lr()[0]
        log_gpu_memory_usage("After offload weights", logger=logger)

        torch.distributed.all_reduce(
            local_scaled_loss, op=torch.distributed.ReduceOp.SUM
        )
        local_scaled_loss /= dp_size
        return {
            "train/loss": local_scaled_loss.item(),
            "train/lr(1e-3)": lr * 1e3,
            "train/grad_norm": grad_norm_value,
        }

    def validation_step(
        self, batch: Mapping
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.fsdp_model.eval()
        with torch.no_grad():
            numerator, denominator = self._weighted_forward_components(batch)
        return numerator.detach().float(), denominator.detach().float()

    def _run_validation(self) -> float:
        """Return one exact weighted mean over the complete validation split."""

        total_numerator = torch.zeros(
            (), device=self.device_name, dtype=torch.float32
        )
        total_denominator = torch.zeros(
            (), device=self.device_name, dtype=torch.float32
        )
        for batch in self.val_dataloader:
            numerator, denominator = self.validation_step(batch)
            total_numerator += numerator
            total_denominator += denominator
        torch.distributed.all_reduce(total_numerator, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(
            total_denominator, op=torch.distributed.ReduceOp.SUM
        )
        if not bool(torch.isfinite(total_denominator)) or total_denominator.item() <= 0:
            raise ValueError("validation split has no positive state weight")
        val_loss = total_numerator / total_denominator
        if not bool(torch.isfinite(val_loss)):
            raise FloatingPointError(f"validation loss is not finite: {val_loss}")
        return float(val_loss.item())

    def save_checkpoint(self, step):
        # save checkpoint
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            # FSDP1 checkpoint saving
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.fsdp_model.state_dict()

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.tokenizer.save_pretrained(path)

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = 0
        last_valid_metric = None
        test_freq = int(self.config.trainer.test_freq)
        save_freq = int(self.config.trainer.save_freq)
        if test_freq < 0 or save_freq < 0:
            raise ValueError("trainer.test_freq and trainer.save_freq must be non-negative")
        if rank == 0:
            print(f"Total training steps: {self.total_training_steps}")

        # The explicit step budget is authoritative. If filtering changes the
        # number of accepted rows, cycle through new deterministic dataset
        # passes instead of silently changing the optimizer-step count.
        epoch = 0
        while global_step < self.total_training_steps:
            self.train_sampler.set_epoch(epoch)
            progress = tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Dataset pass {epoch + 1}",
                disable=rank != 0,
            )
            for data in progress:
                global_step += 1
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = test_freq > 0 and global_step % test_freq == 0
                is_save_step = save_freq > 0 and global_step % save_freq == 0

                if is_last_step or is_valid_step:
                    val_loss = self._run_validation()
                    if rank == 0:
                        metric = {"val/loss": val_loss}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or is_save_step:
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return

            epoch += 1


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()
    del local_rank
    try:
        seed = int(config.trainer.get("seed", 0))
        _seed_everything(seed)
        if rank == 0:
            output_dir = Path(str(config.trainer.default_local_dir))
            output_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(
                config=config,
                f=output_dir / "resolved_config.yaml",
                resolve=True,
            )
        sp_size = int(config.get("ulysses_sequence_parallel_size", 1))
        if sp_size != 1:
            raise NotImplementedError(
                "the audited trainer currently requires ulysses_sequence_parallel_size=1"
            )
        if world_size % sp_size != 0:
            raise ValueError("world size must be divisible by sequence parallel size")
        device_mesh = init_device_mesh(
            device_type=device_name,
            mesh_shape=(world_size,),
            mesh_dim_names=("fsdp",),
        )
        dp_size = world_size // sp_size
        ulysses_device_mesh = init_device_mesh(
            device_type=device_name,
            mesh_shape=(dp_size, sp_size),
            mesh_dim_names=("dp", "sp"),
        )
        from verl.utils import hf_tokenizer

        local_model_path = copy_to_local(
            src=config.model.partial_pretrain, verbose=True
        )
        tokenizer = hf_tokenizer(
            local_model_path, trust_remote_code=config.model.trust_remote_code
        )
        train_dataset = create_sft_dataset(
            config.data.train_files, config.data, tokenizer
        )
        val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)
        trainer = FSDPSFTTrainer(
            config=config,
            device_mesh=device_mesh,
            ulysses_device_mesh=ulysses_device_mesh,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        trainer.fit()
    finally:
        destroy_global_process_group()


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create only the canonical final-assistant-action dataset."""

    return FinalTurnActionDataset(
        parquet_files=data_paths,
        tokenizer=tokenizer,
        config=data_config,
    )


if __name__ == "__main__":
    main()
