# veRL integration contract

This integration is intentionally locked to **veRL 0.4.1**, the version named
by the recovered experiment launcher. The launcher resolves version declarations
from the selected checkout/package, requires an immutable veRL Git revision,
and fails before creating the run directory if either contract is absent.

The canonical dataset returns two separate concepts:

- `target_token_mask[B,T]`: positions of supervised assistant action-content
  tokens, excluding headers, thinking bridges, and turn terminators;
- `state_weight[B]`: sample/state/game weight before token normalization.

For causal labels `input_ids[:, 1:]`, align the mask as
`target_token_mask[:, 1:]`. The historical `[:, :-1]` implementation is an
off-by-one error.

Rows are shuffled uniformly, while `state_weight` defines a generally nonuniform
state/game objective. If a step contains `r` real rows from a training set of
size `D` and total objective weight `W`, compute the shared normalizer `rW/D`
before splitting into microbatches. All-reduce `r` across data-parallel ranks.
Each microbatch contributes its weighted numerator divided by that fixed
normalizer. This is an unbiased stochastic estimate of the full weighted
dataset objective; dividing by the random in-batch weight sum would be a biased
self-normalized estimate and would erase all weights at batch size one.

Backpropagate each contribution directly; do not divide after `backward()` and
do not normalize independently inside each microbatch.

Before any output directory is created, the launcher also validates the whole
realized fixed-budget experiment, not just the current parquet file.  It requires
the schema-v2 `ANNOTATION_PAIR_MANIFEST`, verifies both breadth/depth members and
their common Teacher/state-pool contract, and checks that the current build audit
is exactly the corresponding member.  The frozen state-pool manifest is part of
that contract.  For a Student-state on-policy run, `MODEL_PATH` and the training
Tokenizer must have the same content identity as the behavior Student and
Tokenizer that generated the pool; a model alias is never sufficient.

`omniopd.verl_adapter.uniform_sampling_normalizer` and
`backward_optimizer_batch` are the executable references. The design makes one
optimizer-batch update identical under any microbatch partition, up to
floating-point accumulation order.

The runnable integration is `integrations/verl/fsdp_sft_trainer.py`; the
launcher executes this repository file directly and never imports an unpatched
trainer from an installed veRL checkout. It additionally enforces:

- FP32 model/master parameters, BF16 forward/FSDP parameters, FP32 CE and
  gradient reduction;
- SDPA attention and FSDP1 with `use_orig_params=true`, so FP32 loading is valid
  and PEFT's frozen base/trainable LoRA parameter mixture remains supported;
- a fail-closed check against exceeding the model's native context length;
- an explicit, identical `TOTAL_TRAINING_STEPS` for every compared arm;
- zero-weight distributed and final-batch padding instead of dropping rows;
- `data.truncation=error`. Rollout already reserves target space, so training
  must fail if a target still exceeds the limit rather than silently deleting
  the system/task prefix;
- exact numerator/denominator aggregation over the full validation split;
- no checkpoint resume until optimizer and scheduler state are also saved.

`launch_manifest.json` means only that a launch passed preflight.  It is not proof
of a completed training run.  `completion_manifest.json` is written only after
the distributed trainer and its logging pipeline exit successfully and the
`global_step_<TOTAL_TRAINING_STEPS>` has passed a PEFT LoRA layout check (adapter
config semantics, exactly one `adapter_model.safetensors`, tokenizer config,
launch-matching rank/alpha/`all-linear`, paired LoRA A/B tensors, compatible
shapes/dtypes, valid contiguous safetensors ranges, target-module coverage, and
the absence of full-model weight files),
and the checkpoint, resolved config, and log have all been content-fingerprinted.
The completion manifest uses schema v2. Evaluation and updated-checkpoint M3
scoring require both manifests, reparse the current checkpoint directory, and
reject a launch-only, changed, structurally empty, or non-LoRA directory. This
structural validation does not replace a real PEFT/vLLM load test on the target
cluster.

Sequence parallel and remove-padding are intentionally rejected. Their packed
token boundaries need a separate target-mask audit before they can be enabled.
