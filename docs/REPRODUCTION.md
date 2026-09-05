# Agent OmniOPD 重跑手册

本文给出 `omniopd-v1` 的规范执行顺序。尖括号表示必须由实验者替换的本地资源；
真实 API 密钥只通过环境变量传入，不写入命令历史、配置、日志或 manifest。所有入口
默认拒绝覆盖已有产物；失败后的 partial ledger 应保留用于预算审计，并换新目录重跑。

## 0. 环境与版本

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,api,train]'
python scripts/run_tests.py
python -m ruff check src scripts tests integrations
```

ALFWorld/TextWorld、vLLM 和 CUDA 需按所在集群单独安装。训练集成严格锁定
**veRL 0.4.1**，并要求 veRL checkout 与本代码库都有不可变 Git revision。训练前先验收
fixed-budget 配置：

```bash
python scripts/validate_experiment_pair.py --kind fixed_budget \
  configs/breadth_150x1.yaml configs/depth_50x3.yaml
```

## 1. 一次性冻结 Student state pool

下面参数与两个 fixed-budget 配置中的 `state_pool` 区块一一对应：

```bash
python scripts/collect_state_pool.py \
  --env-config <alfworld_config.yaml> \
  --output-dir <run/state_pool> \
  --behavior-model <served_student_name> \
  --behavior-artifact <immutable_base_or_checkpoint> \
  --behavior-url <student_openai_endpoint> \
  --behavior-thinking-mode disabled \
  --behavior-temperature 0 \
  --state-source student \
  --tokenizer <immutable_student_tokenizer> \
  --limit-games 50 --max-steps 50 \
  --max-context-tokens 4096 --reserve-tokens 256 \
  --behavior-max-tokens 256 --seed 42
```

`state_pool/state_pool.jsonl` 是后续所有选择策略的共同总体。不得为不同实验组分别
rollout。若研究 Teacher-state control，另建独立 pool，并额外传入可追溯的
`--behavior-model-revision`；这属于另一个明确的 state-source intervention。

若运行 uncertainty selection，先对同一 pool 计分并保留清单：

```bash
python scripts/score_entropy.py \
  --state-pool <run/state_pool/state_pool.jsonl> \
  --state-pool-manifest <run/state_pool/manifest.json> \
  --model <immutable_student_model> --tokenizer <immutable_student_tokenizer> \
  --output <run/uncertainty_scores.jsonl>
```

这里的 `--model/--tokenizer` 必须与 state-pool manifest 中记录的 Student 行为模型和
Tokenizer 内容指纹一致；当前入口要求行为模型以一个可加载的合并目录表示。

## 2. 从同一 pool 产生嵌套 breadth/depth selection

```bash
python scripts/select_states.py \
  --state-pool <run/state_pool/state_pool.jsonl> \
  --state-pool-manifest <run/state_pool/manifest.json> \
  --experiment-config configs/breadth_150x1.yaml \
  --output <run/breadth/selection.jsonl>

python scripts/select_states.py \
  --state-pool <run/state_pool/state_pool.jsonl> \
  --state-pool-manifest <run/state_pool/manifest.json> \
  --experiment-config configs/depth_50x3.yaml \
  --output <run/depth/selection.jsonl>
```

同一 seed 为每个 game 生成一条 hash-priority 顺序，因此 depth 的 1 个 state 是 breadth
的 3 个 state 的子集。若要做 uncertainty/top-score 研究，先运行
`scripts/score_entropy.py`，并在单独的 resolved config 中声明
`top_score_per_game`，同时向 selection 入口传入 `--scores` 与
`--scores-manifest`；该确定性设计没有 inclusion probability，不能报告设计加权的总体率。

## 3. 固定预算 Teacher annotation

`M,N,B,max_tokens` 全部从 experiment config 和 sampling profile 读取，命令行不能暗中
覆盖。以 breadth 为例：

```bash
export DEEPSEEK_API_KEY=<secret>
python scripts/annotate_states.py \
  --state-pool <run/state_pool/state_pool.jsonl> \
  --state-pool-manifest <run/state_pool/manifest.json> \
  --selection <run/breadth/selection.jsonl> \
  --selection-manifest <run/breadth/selection.jsonl.manifest.json> \
  --experiment-config configs/breadth_150x1.yaml \
  --teacher-profile configs/teacher_sampling_nonthinking.yaml \
  --output-dir <run/breadth/annotation> \
  --teacher-model <provider_model_name> \
  --teacher-model-revision <provider_snapshot_label> \
  --teacher-tokenizer <immutable_teacher_tokenizer> \
  --teacher-context-window <context_window>
```

对 depth 重复一次，只替换 config、selection 和输出目录。两组完成后验收真实调用与
嵌套集合，而不只检查计划配置：

```bash
python scripts/validate_experiment_pair.py --kind annotation_runs \
  <run/breadth/annotation/manifest.json> \
  <run/depth/annotation/manifest.json>
```

必须满足 `actual_teacher_api_calls = M×N = B`。API error、空输出、解析失败与
不在 admissible set 的动作均占用一次预算并保留为 invalid；没有免费 retry。

## 4. 构建 action-only 数据

```bash
python -m omniopd.cli.build_data \
  --input <run/breadth/annotation/corrections.jsonl> \
  --correction-manifest <run/breadth/annotation/manifest.json> \
  --experiment-config configs/breadth_150x1.yaml \
  --output <run/breadth/train.parquet> \
  --validation-output <run/breadth/validation.parquet>
```

depth 同理。split unit 是 game；默认 `game_state_mean` 的每个保留 game 总权重为 1，
每 state 内所有有效 Teacher samples 再平分该 state 权重。该目标明确条件于
`Teacher-valid`，audit 会报告 state/sample acceptance 与完全丢失的 game。

## 5. 固定 optimizer-step 训练

对配置中 `[7,42,123,2024]` 的每个 seed，分别训练 breadth 与 depth。下面只展示一个
seed；所有数值必须与对应 config 一致：

```bash
VERL_ROOT=<verl-0.4.1-checkout> \
MODEL_PATH=<immutable_base_model> \
TRAIN_FILES=<run/breadth/train.parquet> \
VAL_FILES=<run/breadth/validation.parquet> \
OUTPUT_DIR=<checkpoints/breadth_seed7> \
NUM_GPUS=4 TOTAL_TRAINING_STEPS=102 TRAINING_SEED=7 \
EXPERIMENT_CONFIG=configs/breadth_150x1.yaml \
TRAIN_AUDIT=<run/breadth/train.parquet.audit.json> \
LR=2e-5 TRAIN_BSZ=4 MICRO_BSZ=1 MAX_LENGTH=4096 \
MODEL_DTYPE=fp32 TRAINING_DTYPE=bf16 LORA_RANK=16 LORA_ALPHA=32 \
bash scripts/run_verl_train.sh
```

每个 seed 的两组 launch manifest 必须成对验收：

```bash
python scripts/validate_experiment_pair.py --kind training_runs \
  <checkpoints/breadth_seed7/launch_manifest.json> \
  <checkpoints/depth_seed7/launch_manifest.json>
```

trainer 使用 action-content-only mask；assistant header、Qwen generation bridge、EOT
均不计入动作 loss。uniform row sampling 下，optimizer batch 的共享 normalizer 为
`rW/D`，microbatch 只累加 numerator；分布式补齐样本权重为 0。

## 6. 冻结评测列表并配对评测

先创建独立于训练 games 的 ID/OOD game list：

```bash
python scripts/freeze_game_list.py \
  --env-config <alfworld_config.yaml> \
  --split eval_out_of_distribution --limit-games <G_eval> --seed 2026 \
  --exclude-jsonl <run/state_pool/state_pool.jsonl> \
  --output <eval/ood_games.json>
```

评测包装器强制区分 checkpoint 的 `TRAINING_SEED` 与所有组共享的 `ROLLOUT_SEED`：

```bash
CUDA_DEVICE=0 BASE_MODEL=<base_model> BASE_SERVED_MODEL=omniopd-base \
LORA_PATH=<adapter> \
TRAINING_MANIFEST=<checkpoints/breadth_seed7/launch_manifest.json> \
SERVED_MODEL=omniopd-student PORT=<port> \
EVAL_OUTPUT=<eval/breadth_seed7_ood.json> \
ENV_CONFIG=<alfworld_config.yaml> TOKENIZER_PATH=<tokenizer> \
EVAL_SPLIT=eval_out_of_distribution LIMIT_GAMES=<G_eval> \
GAME_LIST=<eval/ood_games.json> \
GAME_LIST_MANIFEST=<eval/ood_games.json.manifest.json> \
TRAINING_SEED=7 ROLLOUT_SEED=2026 \
MAX_STEPS=50 MAX_CONTEXT_TOKENS=4096 RESERVE_TOKENS=256 MAX_TOKENS=256 \
VLLM_MAX_MODEL_LEN=8192 bash scripts/run_vllm_eval.sh
```

每个 checkpoint 在隔离的服务进程中加载，但所有比较组必须使用同一个稳定且不同于
base alias 的 `SERVED_MODEL` LoRA 别名（例如 `omniopd-student`）；真实 checkpoint 身份由
`MODEL_PATH/LORA_PATH` 的内容指纹记录，不能靠不同服务别名表达。这样聚合器既能拒绝
服务端返回了意外模型，也不会把别名差异误当成 rollout regime 差异。

每一 arm 汇总四个训练 seed：

```bash
python scripts/aggregate_evaluations.py \
  --evaluation 7=<eval/arm_seed7.json> \
  --evaluation 42=<eval/arm_seed42.json> \
  --evaluation 123=<eval/arm_seed123.json> \
  --evaluation 2024=<eval/arm_seed2024.json> \
  --output <eval/arm_seed_game.json>
```

最后运行带 manifest 约束的 paired hierarchical bootstrap：

```bash
omniopd-bootstrap \
  --treatment <eval/treatment_seed_game.json> \
  --treatment-manifest <eval/treatment_seed_game.json.manifest.json> \
  --control <eval/control_seed_game.json> \
  --control-manifest <eval/control_seed_game.json.manifest.json> \
  --replicates 50000 --confidence 0.95 --seed 42 \
  --output <eval/paired_bootstrap.json>
```

该入口要求两组的 training-seed set、逐 game 键、game list、环境、prompt、Tokenizer、
代码 revision、rollout seed、thinking、generation limit 与 provider fingerprint 完全一致。

## 7. 机制分析

### M1：surrogate-gradient alignment

```bash
python scripts/analyze_m1_gradients.py \
  --base-model <base> --tokenizer <tokenizer> \
  --reference <preregistered_game_disjoint_probe.parquet> \
  --reference-audit <probe.parquet.audit.json> \
  --group A1=<a1_all_valid_rows.parquet> \
  --group A3=<a3_all_valid_rows.parquet> \
  --group-audit A1=<a1_all_valid_rows.parquet.audit.json> \
  --group-audit A3=<a3_all_valid_rows.parquet.audit.json> \
  --output-dir <analysis/m1>
```

三份 action table 均须由规范 builder 以“不划 validation”的方式构建，并由各自 audit
绑定 correction manifest。reference games 必须与 correction games 不相交。结果只能称
held-out Teacher-CE surrogate alignment，不能称真实成功率梯度 (G_*)。

### M2：game-balanced representativeness

```bash
python scripts/analyze_m2_representativeness.py \
  --state-pool <run/state_pool/state_pool.jsonl> \
  --state-pool-manifest <run/state_pool/manifest.json> \
  --group A1=<a1_corrections.jsonl> --group A3=<a3_corrections.jsonl> \
  --group-manifest A1=<a1_annotation/manifest.json> \
  --group-manifest A3=<a3_annotation/manifest.json> \
  --output <analysis/m2.json>
```

这是边际分布诊断，不是 correction utility 的因果检验。

### M3：冻结邻居后的 action-imitation transfer

先为 A1/A3 各构建一份**不划 validation、含全部有效样本**的数据及 audit，然后冻结
所有实验 selection 的并集、跨 game 邻居和 score IDs：

```bash
python scripts/prepare_m3_panel.py \
  --state-pool <run/state_pool/state_pool.jsonl> \
  --state-pool-manifest <run/state_pool/manifest.json> \
  --source A1=<a1_all_rows.jsonl> --source A3=<a3_all_rows.jsonl> \
  --source-audit A1=<a1_all_rows.audit.json> \
  --source-audit A3=<a3_all_rows.audit.json> \
  --selection A1=<a1_selection.jsonl> --selection A3=<a3_selection.jsonl> \
  --selection-manifest A1=<a1_selection.manifest.json> \
  --selection-manifest A3=<a3_selection.manifest.json> \
  --output-dir <analysis/m3_panel> --neighbors 10 --minimum-neighbors 5
```

对每个 group 的 frozen rows 分别计算共同 base 与所属 adapter 的分数（四次运行必须使用
同一 base、Tokenizer 和 dtype）：

```bash
python scripts/score_action_logp.py \
  --data <analysis/m3_panel/score_rows_A1.jsonl> \
  --base-model <base> --tokenizer <tokenizer> --model-dtype bf16 \
  --output <a1_base_scores.jsonl>
python scripts/score_action_logp.py \
  --data <analysis/m3_panel/score_rows_A1.jsonl> \
  --base-model <base> --adapter <a1_final_adapter> \
  --training-manifest <a1_checkpoint/launch_manifest.json> \
  --tokenizer <tokenizer> --model-dtype bf16 \
  --output <a1_updated_scores.jsonl>
```

更新后计分必须证明 adapter 是相应规范训练 manifest 声明的最终 `global_step`，且训练
base 与计分 base 内容一致。A3 对应重复上述两条命令，然后：

```bash
python scripts/assemble_m3.py \
  --preparation-manifest <analysis/m3_panel/manifest.json> \
  --score-pair A1=<a1_base_scores.jsonl>,<a1_updated_scores.jsonl> \
  --score-pair A3=<a3_base_scores.jsonl>,<a3_updated_scores.jsonl> \
  --output <analysis/m3_rows.jsonl>

python scripts/analyze_m3.py \
  --input <analysis/m3_rows.jsonl> \
  --input-manifest <analysis/m3_rows.jsonl.manifest.json> \
  --replicates 5000 --seed 42 --output <analysis/m3.json>
```

这里的 (S_i,T_i) 是 whole-checkpoint 的 action-imitation log-prob shift，不等同于
单条 correction 的任务效用或真实 ΔJ。

### SAGE：盲 intervention judge

```bash
python scripts/count_state_population.py \
  --state-pool <run/state_pool/state_pool.jsonl> \
  --state-pool-manifest <run/state_pool/manifest.json> \
  --output <analysis/population_counts.json>

python scripts/judge_sage.py \
  --corrections <corrections.jsonl> \
  --correction-manifest <annotation/manifest.json> \
  --output-dir <analysis/sage_labels> \
  --judge-profile configs/sage_judge_thinking.yaml \
  --judge-model <model> --judge-model-revision <snapshot> \
  --judge-tokenizer <tokenizer> --judge-context-window <window> \
  --judge-budget <number_of_student_valid_states>

python scripts/analyze_sage.py \
  --labels <analysis/sage_labels/labels.jsonl> \
  --labels-manifest <analysis/sage_labels/manifest.json> \
  --population-counts <analysis/population_counts.json> \
  --population-manifest <analysis/population_counts.json.manifest.json> \
  --output-dir <analysis/sage>
```

Judge 看不到 Teacher action、entropy、选择分数或组别。Student-invalid state 直接进入
deterministic Strong；API/解析失败保留为 missing。若 selection 是 deterministic top-k，
报告只保留 observed-sample 描述量，总体 HT estimate 明确为不可识别。

### Position：单步动作后果

```bash
python scripts/run_position_counterfactual.py \
  --corrections <corrections.jsonl> \
  --correction-manifest <annotation/manifest.json> \
  --env-config <alfworld_config.yaml> \
  --student-model <served_student> --student-artifact <checkpoint> \
  --tokenizer <tokenizer> --inference-runtime <vllm:version> \
  --rollout-seed 2026 --output-dir <analysis/position>
```

Student/Teacher 两个 intervention branch 都从同一 prefix 独立 replay，并由同一个冻结
Student continuation policy 接管；(a_T=a_S) 时 trace 必须完全一致。该量是单步
consequentiality，不是完整训练 utility。

## 8. 结果解释与重跑边界

- 旧 Word 代码及旧 checkpoint 只保存在 `legacy/` 供追溯，不能与 `omniopd-v1` 新结果混合；
- P0 修复改变了 state、prompt、Teacher query、loss mask 与 optimizer update，确认性结果
  必须从 state pool 开始重跑；
- `game_state_mean` 只在 Teacher-valid 条件分布内实现 game balance；必须同时报告 attrition；
- 没有真实 DeepSeek 凭据、ALFWorld 数据、目标模型、veRL 0.4.1 checkout 和 GPU 时，仓库的
  本地测试只能验证逻辑契约，不能替代端到端科学复现。
