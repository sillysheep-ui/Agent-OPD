# Agent OmniOPD 规范重跑手册

本手册描述 `omniopd-v1` 当前真正可闭环的确认性主链：同一 Student
state pool 上的 fixed-budget breadth `150×1` 与 depth `50×3` 对照。尖括号表示
实验者必须替换的本地资源；不得把它们原样当作命令执行。所有入口默认拒绝
覆盖产物。失败时保留 partial API ledger 供预算审计，换新输出目录重跑；
不允许删除 invalid/error attempt 后“补打”免费请求。

## 0. 冻结代码、环境与协议

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,api,train]'
python scripts/run_tests.py
python -m pytest -q
python -m ruff check src scripts tests integrations
python -m compileall -q src scripts tests integrations
```

开始采集前先提交本仓库，确保 `git status --short` 为空；之后不得修改
`src/`、`scripts/`、`integrations/`、`configs/`、`pyproject.toml` 或 `README.md`。
各阶段会绑定同一 Git revision 和 canonical code-tree hash，中途修改会 fail closed。
若使用不含 `.git` 的交付压缩包，解压后必须先在该目录初始化 Git、加入全部交付文件并
创建一次冻结提交，再执行测试和实验；更推荐从对应提交克隆。压缩包内的
`docs/CODE_INVENTORY.json` 用于核对发布时的实现提交与逐文件 SHA256，新建的本地冻结
提交则作为本次实验 manifest 的 revision。

ALFWorld/TextWorld、vLLM 和 CUDA 需在目标集群安装。训练另外要求一个干净的
**veRL 0.4.1** checkout；本仓库与 veRL checkout 都必须有不可变 revision。先验收
预注册配置：

```bash
python scripts/validate_experiment_pair.py --kind fixed_budget \
  configs/breadth_150x1.yaml configs/depth_50x3.yaml
```

## 1. 一次性冻结 Student state pool

确认性 Student-state 采集必须通过 wrapper 启动本地 vLLM，并在服务仍是 wrapper
存活子进程时校验实际 argv、端口、模型、Tokenizer 和 runtime：

```bash
CUDA_DEVICE=0 \
MODEL_PATH=<complete_behavior_student_dir> \
SERVED_MODEL=<stable_behavior_student_alias> \
PORT=<free_loopback_port> \
OUTPUT_DIR=<run/state_pool> \
ENV_CONFIG=<alfworld_config.yaml> \
TOKENIZER_PATH=<same_complete_behavior_student_dir> \
LIMIT_GAMES=50 GAME_ORDER_SEED=42 ENVIRONMENT_SEED=314159 \
MAX_STEPS=50 MAX_CONTEXT_TOKENS=4096 RESERVE_TOKENS=256 MAX_TOKENS=256 \
VLLM_MAX_MODEL_LEN=8192 \
bash scripts/run_vllm_state_pool.sh
```

`MODEL_PATH` 与 `TOKENIZER_PATH` 必须是同一个可完整加载的 Hugging Face Student
目录，不是“base 目录 + 独立 Tokenizer/LoRA”的组合。wrapper 写出：

- `<run/state_pool>/state_pool.jsonl`；
- `<run/state_pool>/manifest.json`；
- `<run/state_pool>.service_manifest.json`。

后两个实验臂只能从这份 pool 选择，不得分别 rollout。game 顺序 seed 与
TextWorld environment seed 是两个独立字段；manifest 还绑定每个 trial 的字节和
ALFWorld/TextWorld/Torch/Transformers/OpenAI 运行时版本。

若只做选择机制的探索性分析，可对同一 pool 计算 uncertainty：

```bash
python scripts/score_entropy.py \
  --state-pool <run/state_pool/state_pool.jsonl> \
  --state-pool-manifest <run/state_pool/manifest.json> \
  --model <complete_behavior_student_dir> \
  --tokenizer <same_complete_behavior_student_dir> \
  --output <run/uncertainty_scores.jsonl>
```

计分模型和 Tokenizer 内容指纹必须精确等于生成 pool 的 behavior Student。输出 manifest
还绑定pool、代码、模型和Tokenizer；每一行必须覆盖对应state的全部 admissible actions，
后续消费者会从这些有限 action log scores 重算 entropy，而不是只信任 `score` 字段。

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

同一 seed 在每个 game 内生成同一条 hash-priority 顺序，因此 depth 的50个 states
逐 game 嵌套于 breadth 的150个 states。这是 within-game SRSWOR：对 game (g)，
入样概率为 (m/T_g)。

`top_score_per_game` 可用于明确声明的选择机制研究，但 deterministic top-k 没有
已知的非零 inclusion probability；因此不能把它的观测样本率伪写成有限总体
Horvitz–Thompson 估计。使用该策略时，必须把同一份 `--scores` 和
`--scores-manifest` 同时传给 `select_states.py` 与 `annotate_states.py`；两个入口都会
验证完整pool覆盖、behavior Student/Tokenizer身份，并重建逐game top-k及稳定并列顺序。

## 3. 固定预算 Teacher annotation 与整对绑定

通过受保护的运行环境注入 `DEEPSEEK_API_KEY`，不把密钥写入脚本、命令行、
日志或 manifest。以 breadth 为例：

```bash
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
  --teacher-context-window <teacher_context_window>
```

对 depth 只替换 config、selection 和输出目录。\(M,N,B,\text{max tokens}\) 来自已解析配置，
命令行不能暗中覆盖。每个 API error、空输出、解析失败或不在 admissible set 的
动作都占用一次 (B)。两臂都完成后生成唯一的 schema-v2 整对桥接文件：

```bash
python scripts/validate_experiment_pair.py \
  --kind annotation_runs \
  --output <run/annotation_pair.json> \
  <run/breadth/annotation/manifest.json> \
  <run/depth/annotation/manifest.json>
```

该入口不只验证 (B=MN=150) 和嵌套选择，还绑定完整 Teacher、behavior
Student、state pool、code、provider、prompt、Tokenizer、selection 和 invalid-policy 契约。
两个训练臂必须绑定同一份 `<run/annotation_pair.json>` 的相同 SHA256。

## 4. 构建 action-only 训练/验证数据

```bash
python -m omniopd.cli.build_data \
  --input <run/breadth/annotation/corrections.jsonl> \
  --correction-manifest <run/breadth/annotation/manifest.json> \
  --experiment-config configs/breadth_150x1.yaml \
  --output <run/breadth/train.parquet> \
  --validation-output <run/breadth/validation.parquet>
```

depth 同理。split unit 是 game，不是 row/state。`game_state_mean` 对每个至少有一次
Teacher-valid draw 的保留 state 等权，对每个保留 game 等权；每 state 内的
有效 draws 再平分该 state 权重。audit 必须报告 state/sample acceptance 以及
因所有 draws 无效而完全丢失的 games。

模型学习的 action 分布是
(q_T^V(a\mid s)=P(A=a\mid s,V_T=1))；invalid attempts 只进预算/attrition ledger，
不伪造 action target。因为保留概率为 (P(K_s>0\mid s))，改变 (N) 会改变
进入训练集的 state 分布。

## 5. 固定 optimizer-step 训练

对 `[7,42,123,2024]` 的每个 training seed，分别训练 breadth 和 depth。以
breadth/seed 7 为例：

```bash
VERL_ROOT=<clean_verl_0.4.1_checkout> \
MODEL_PATH=<same_complete_behavior_student_dir> \
TRAIN_FILES=<run/breadth/train.parquet> \
VAL_FILES=<run/breadth/validation.parquet> \
OUTPUT_DIR=<checkpoints/breadth_seed7> \
NUM_GPUS=4 TOTAL_TRAINING_STEPS=102 TRAINING_SEED=7 \
EXPERIMENT_CONFIG=configs/breadth_150x1.yaml \
TRAIN_AUDIT=<run/breadth/train.parquet.audit.json> \
ANNOTATION_PAIR_MANIFEST=<run/annotation_pair.json> \
LR=2e-5 TRAIN_BSZ=4 MICRO_BSZ=1 MAX_LENGTH=4096 \
MODEL_DTYPE=fp32 TRAINING_DTYPE=bf16 LORA_RANK=16 LORA_ALPHA=32 \
bash scripts/run_verl_train.sh
```

`MODEL_PATH` 必须精确等于 state-pool manifest 的完整 behavior Student，训练
Tokenizer 也从该同一目录加载。wrapper 在启动前校验数据/audit/config/pair、两个
干净工作树、veRL 0.4.1、dtype、batch/microbatch 和 fixed step 契约。只有训练正常
退出，且 final step 是带唯一 `adapter_model.safetensors`、不含完整模型权重、带Tokenizer配置、rank/alpha/
`all-linear`匹配 launch、A/B tensors成对且shape/dtype/offset和target覆盖有效的 PEFT
LoRA 目录，同时 resolved config 和 `train.log` 完整时，才会生成：

- `<checkpoints/...>/launch_manifest.json`；
- `<checkpoints/...>/completion_manifest.json`；
- `<checkpoints/...>/global_step_102/`。

同一 seed 的两臂训练完成后，再验收 launch 对：

```bash
python scripts/validate_experiment_pair.py --kind training_runs \
  <checkpoints/breadth_seed7/launch_manifest.json> \
  <checkpoints/depth_seed7/launch_manifest.json>
```

这里的 `training_runs` 验证比较设计与 launch 公平性，不等于证明训练完成；评测与M3仍
必须分别提供并验证两个 `completion_manifest.json`，且会重新解析当前 checkpoint 实体。

trainer 使用 action-content-only mask；assistant header、Qwen generation bridge、thinking
和 EOT 均不计入 action loss。uniform-row sampling 下，optimizer batch 共享固定
normalizer (rW/D)；microbatch 只累加 numerator，分布式补齐 row 权重为 0。

## 6. 冻结 game list、评测和分层推断

先冻结与训练 games 不重叠的 ID/OOD 列表。选择 seed 与环境 seed 分开：

```bash
python scripts/freeze_game_list.py \
  --env-config <alfworld_config.yaml> \
  --split eval_out_of_distribution \
  --limit-games <G_eval> \
  --seed 2026 \
  --environment-seed <eval_environment_seed> \
  --exclude-jsonl <run/state_pool/state_pool.jsonl> \
  --output <eval/ood_games.json>
```

每个 checkpoint 必须由 wrapper 在隔离的本地 vLLM 子进程中加载。以
breadth/seed 7 为例：

```bash
CUDA_DEVICE=0 \
BASE_MODEL=<same_complete_behavior_student_dir> \
BASE_SERVED_MODEL=<stable_base_alias> \
LORA_PATH=<checkpoints/breadth_seed7/global_step_102> \
TRAINING_MANIFEST=<checkpoints/breadth_seed7/launch_manifest.json> \
TRAINING_COMPLETION_MANIFEST=<checkpoints/breadth_seed7/completion_manifest.json> \
SERVED_MODEL=<stable_lora_alias_distinct_from_base_alias> \
PORT=<free_loopback_port> \
EVAL_OUTPUT=<eval/breadth_seed7_ood.json> \
ENV_CONFIG=<alfworld_config.yaml> \
TOKENIZER_PATH=<same_complete_behavior_student_dir> \
EVAL_SPLIT=eval_out_of_distribution LIMIT_GAMES=<G_eval> \
GAME_LIST=<eval/ood_games.json> \
GAME_LIST_MANIFEST=<eval/ood_games.json.manifest.json> \
TRAINING_SEED=7 ROLLOUT_SEED=2026 ENVIRONMENT_SEED=<eval_environment_seed> \
MAX_STEPS=50 MAX_CONTEXT_TOKENS=4096 RESERVE_TOKENS=256 MAX_TOKENS=256 \
VLLM_MAX_MODEL_LEN=8192 \
bash scripts/run_vllm_eval.sh
```

各比较组使用相同的稳定 base/LoRA alias，但两个 alias 必须不同。真实
checkpoint 身份由 base/adapter 内容指纹、launch/completion manifest 和存活服务
attestation 表达，不靠文件名或 alias 猜测。training seed、所有组共享的 rollout
seed、environment seed 和后续 bootstrap seed 是四个不同字段。

每一 arm 汇总四个 training seeds：

```bash
python scripts/aggregate_evaluations.py \
  --evaluation 7=<eval/arm_seed7.json> \
  --evaluation 42=<eval/arm_seed42.json> \
  --evaluation 123=<eval/arm_seed123.json> \
  --evaluation 2024=<eval/arm_seed2024.json> \
  --output <eval/arm_seed_game.json>
```

处置效应方向是 `breadth_minus_depth`，因此 treatment/control 顺序不能反过来：

```bash
python -m omniopd.cli.bootstrap \
  --treatment <eval/breadth_seed_game.json> \
  --treatment-manifest <eval/breadth_seed_game.json.manifest.json> \
  --control <eval/depth_seed_game.json> \
  --control-manifest <eval/depth_seed_game.json.manifest.json> \
  --replicates 50000 --confidence 0.95 --seed 42 \
  --output <eval/paired_bootstrap.json>
```

默认同时重采 training seed 和共享 game 簇，因此至少需2个独立 training seeds
与2个 paired games。`--fixed-seeds` 只能在明确把 estimand 限定为“条件于已给
checkpoints”时使用。

## 7. 机制分析

### 7.1 M1：held-out Teacher-CE surrogate-gradient alignment

三份 action table 都必须由规范 builder 生成。M1 专用表不划 validation，audit 必须
明确 `split=null, validation_rows=0`；它们不是用来训练 checkpoint 的主链 train
split。reference 的 validity 前 selected games 必须与两组不相交，但 Teacher 完整
contract 和 behavior Student 必须相同。以 fixed-budget 两组为例：

```bash
python scripts/analyze_m1_gradients.py \
  --base-model <same_complete_behavior_student_dir> \
  --tokenizer <same_complete_behavior_student_dir> \
  --reference <heldout_reference_all_valid_rows.parquet> \
  --reference-audit <heldout_reference_all_valid_rows.parquet.audit.json> \
  --group breadth_150x1=<breadth_all_valid_rows.parquet> \
  --group depth_50x3=<depth_all_valid_rows.parquet> \
  --group-audit breadth_150x1=<breadth_all_valid_rows.parquet.audit.json> \
  --group-audit depth_50x3=<depth_all_valid_rows.parquet.audit.json> \
  --annotation-pair <run/annotation_pair.json> \
  --output-dir <analysis/m1>
```

两组必须精确复现 pair 的两个 members，并先限制到两组共同的 Teacher-valid
retained-game support。同一 state 的多个有效 draws 先平均 gradient，再计算 dot、norm
和 cosine 等非线性指标。零范数 cosine 是不可识别，不报为0。M1 只能称
held-out Teacher-CE surrogate alignment，不是成功率目标的真实 (G_*^\top G_q)。

### 7.2 M2：共同 retained-game support 上的边际表征性

```bash
python scripts/analyze_m2_representativeness.py \
  --state-pool <run/state_pool/state_pool.jsonl> \
  --state-pool-manifest <run/state_pool/manifest.json> \
  --group breadth=<run/breadth/annotation/corrections.jsonl> \
  --group depth=<run/depth/annotation/corrections.jsonl> \
  --group-manifest breadth=<run/breadth/annotation/manifest.json> \
  --group-manifest depth=<run/depth/annotation/manifest.json> \
  --replicates 5000 --seed 42 \
  --output <analysis/m2.json>
```

所有组必须同 pool/code/完整 Teacher contract。比较 support 是所有组 Teacher-valid
retained games 的交集，且至少2个 games。JS 对每 game empirical mass 等权，CI 用
paired game-cluster bootstrap。这只是共同 support 上的边际分布诊断，不是
correction utility 的因果检验。

### 7.3 M3：checkpoint×panel action-imitation transfer（分析器接口）

M3 必须有两个 (N=1) 的 A1/A3 已完成 checkpoints。每个 source 不是“不划
validation 的全部样本”，而是对应 checkpoint **真实使用的 training split**；audit
必须精确绑定 `training_output_sha256`，并证明存在独立 validation games。冻结 panel：

```bash
python scripts/prepare_m3_panel.py \
  --state-pool <run/state_pool/state_pool.jsonl> \
  --state-pool-manifest <run/state_pool/manifest.json> \
  --source A1=<a1_actual_train_split.parquet> \
  --source A3=<a3_actual_train_split.parquet> \
  --source-audit A1=<a1_actual_train_split.parquet.audit.json> \
  --source-audit A3=<a3_actual_train_split.parquet.audit.json> \
  --selection A1=<a1_selection.jsonl> \
  --selection A3=<a3_selection.jsonl> \
  --selection-manifest A1=<a1_selection.jsonl.manifest.json> \
  --selection-manifest A3=<a3_selection.jsonl.manifest.json> \
  --output-dir <analysis/m3_panel> \
  --neighbors 10 --minimum-neighbors 5
```

`--selection/--selection-manifest` 还必须包含所有需从候选邻居中排除的设计。然后计算
2个 BASE cells 和4个 updated cells，总共6格。Tokenizer 参数应省略（默认为 base
目录），或给出与 base 完全相同的路径：

```bash
python scripts/score_action_logp.py \
  --data <analysis/m3_panel/score_rows_A1.jsonl> --base-model <base> \
  --checkpoint-group BASE --panel-group A1 --output <scores/base_A1.jsonl>
python scripts/score_action_logp.py \
  --data <analysis/m3_panel/score_rows_A3.jsonl> --base-model <base> \
  --checkpoint-group BASE --panel-group A3 --output <scores/base_A3.jsonl>

python scripts/score_action_logp.py \
  --data <analysis/m3_panel/score_rows_A1.jsonl> --base-model <base> \
  --adapter <a1_final_checkpoint> --training-manifest <a1_launch_manifest.json> \
  --training-completion-manifest <a1_completion_manifest.json> \
  --checkpoint-group A1 --panel-group A1 --output <scores/A1_A1.jsonl>
python scripts/score_action_logp.py \
  --data <analysis/m3_panel/score_rows_A3.jsonl> --base-model <base> \
  --adapter <a1_final_checkpoint> --training-manifest <a1_launch_manifest.json> \
  --training-completion-manifest <a1_completion_manifest.json> \
  --checkpoint-group A1 --panel-group A3 --output <scores/A1_A3.jsonl>
python scripts/score_action_logp.py \
  --data <analysis/m3_panel/score_rows_A1.jsonl> --base-model <base> \
  --adapter <a3_final_checkpoint> --training-manifest <a3_launch_manifest.json> \
  --training-completion-manifest <a3_completion_manifest.json> \
  --checkpoint-group A3 --panel-group A1 --output <scores/A3_A1.jsonl>
python scripts/score_action_logp.py \
  --data <analysis/m3_panel/score_rows_A3.jsonl> --base-model <base> \
  --adapter <a3_final_checkpoint> --training-manifest <a3_launch_manifest.json> \
  --training-completion-manifest <a3_completion_manifest.json> \
  --checkpoint-group A3 --panel-group A3 --output <scores/A3_A3.jsonl>
```

```bash
python scripts/assemble_m3.py \
  --preparation-manifest <analysis/m3_panel/manifest.json> \
  --base-score A1=<scores/base_A1.jsonl> \
  --base-score A3=<scores/base_A3.jsonl> \
  --updated-score A1,A1=<scores/A1_A1.jsonl> \
  --updated-score A1,A3=<scores/A1_A3.jsonl> \
  --updated-score A3,A1=<scores/A3_A1.jsonl> \
  --updated-score A3,A3=<scores/A3_A3.jsonl> \
  --output <analysis/m3_rows.jsonl>

python scripts/analyze_m3.py \
  --input <analysis/m3_rows.jsonl> \
  --input-manifest <analysis/m3_rows.jsonl.manifest.json> \
  --replicates 5000 --seed 42 --output <analysis/m3.json>
```

**发布边界：**当前仓库的规范 pair/training launcher 只生成 breadth `N=1` 与
depth `N=3` 的 fixed-budget checkpoints，尚无另一对 (N=1) A1/A3 state-selection
pair launcher。所以上述是精确的分析器接口和未来验收契约，不是当前可从本包
生成 M3 确认性结果的主链。当前 M3 也未实现多 training-seed 结果的最终
汇总。不得把“入口已实现”写成“确认性 M3 已复现”。

### 7.4 SAGE：去重并集上的盲 intervention judge（A1/A3 分析接口）

先冻结每个 target game 的有限 state 总体大小：

```bash
python scripts/count_state_population.py \
  --state-pool <run/state_pool/state_pool.jsonl> \
  --state-pool-manifest <run/state_pool/manifest.json> \
  --output <analysis/population_counts.json>
```

每个分组必须 (N=1)、同 pool/Teacher/code、实验 ID 不同。输入顺序成对重复：

```bash
python scripts/judge_sage.py \
  --corrections <a1_corrections.jsonl> \
  --correction-manifest <a1_annotation/manifest.json> \
  --corrections <a3_corrections.jsonl> \
  --correction-manifest <a3_annotation/manifest.json> \
  --output-dir <analysis/sage_labels> \
  --judge-profile configs/sage_judge_thinking.yaml \
  --judge-model <judge_model> \
  --judge-model-revision <judge_snapshot> \
  --judge-tokenizer <immutable_judge_tokenizer> \
  --judge-context-window <judge_context_window> \
  --judge-budget <unique_student_valid_states_in_union>

python scripts/analyze_sage.py \
  --labels <analysis/sage_labels/labels.jsonl> \
  --labels-manifest <analysis/sage_labels/manifest.json> \
  --population-counts <analysis/population_counts.json> \
  --population-manifest <analysis/population_counts.json.manifest.json> \
  --contrast-reference A1 --contrast-comparison A3 \
  --replicates 5000 --seed 42 \
  --output-dir <analysis/sage>
```

重叠 state 只盲评一次，但保留每个组 membership 各自的 Teacher draw、(D)、(V_T)
和 inclusion probability。`judge-budget` 是去重并集中 Student-valid unique states 数；
Student-invalid state 直接记 technical Strong 且不调 API，judge error 保持 missing。
主报
(U_g=P(V_T=1,D=1,I=\mathrm{Skip}\mid V_S=1,g))，并报 (P(I\mid D))、
(P(D\mid I))、(P(V_T\mid I)) 及全 selected-state secondary。总体 HT/design-weighted ratio
仅在所有非零 inclusion probabilities 已知且目标 games 完整覆盖时可识别；
game-cluster interval 至少需2个 games，并且只传播 game 簇不确定性。

### 7.5 Position：单步动作后果

Position 必须启动生成 state pool 的同一完整 behavior Student，不是训练后
adapter：

```bash
CUDA_DEVICE=0 \
MODEL_PATH=<same_complete_behavior_student_dir> \
SERVED_MODEL=<same_behavior_student_alias> \
PORT=<free_loopback_port> \
OUTPUT_DIR=<analysis/position> \
CORRECTIONS=<corrections.jsonl> \
CORRECTION_MANIFEST=<annotation/manifest.json> \
STATE_POOL=<run/state_pool/state_pool.jsonl> \
STATE_POOL_MANIFEST=<run/state_pool/manifest.json> \
ENV_CONFIG=<alfworld_config.yaml> \
TOKENIZER_PATH=<same_complete_behavior_student_dir> \
MAX_STEPS=50 MAX_CONTEXT_TOKENS=4096 RESERVE_TOKENS=256 MAX_TOKENS=256 \
ROLLOUT_SEED=2026 BOOTSTRAP_SEED=42 BOOTSTRAP_REPLICATES=50000 \
CONFIDENCE=0.95 VLLM_MAX_MODEL_LEN=8192 \
bash scripts/run_vllm_position.sh
```

Student/Teacher intervention branch 都从同一 `full_messages` prefix 独立 replay，再由同一
冻结 Student continuation policy 接管；(a_T=a_S) 时两支 trace 必须逐步一致。同 state
多个有效 Teacher draws 先求 (C_t) 均值。主 estimand 是
(E[C\mid D=1,V_T=1])，unconditional 只作 secondary。总体 HT 只在已知 inclusion
probability 时可识别；top-k 只报 descriptive。runtime、environment/game 字节、context
和 horizon 必须与 state-pool manifest 一致，且 interval 至少需2个 games。

## 8. 公式口径与重跑边界

- Teacher 实际可学习动作分布是解析且 admissible 后的 (q_T^V)，不是未条件化
  的 provider 原始分布。
- 模型所见状态是 (\tau_\Lambda(z_t))，而不是用于 replay/provenance 的无限长
  `full_messages`。
- 在每 game 内从 (T_g) 个 states 中 SRSWOR 选 (m) 个时，条件设计方差口径为
  \[
  \frac1{G^2}\sum_g\left[
  \left(1-\frac m{T_g}\right)\frac{S^2_{\mu,g}}m
  +\frac{\bar\Sigma_{T,g}}{mN}
  \right].
  \]
  若把 games 视为 superpopulation，还必须加 between-game 方差；不得把 iid 简式
  当成当前分层有限总体设计的精确方差。
- 旧 Word 代码及旧 checkpoints 只在 `legacy/` 中用于追溯。P0 修复改变了 state、
  prompt、Teacher query、loss mask 和 optimizer update，确认性结果必须从新 pool
  开始重跑。
- `student_state_control.yaml` 与 `teacher_state_control.yaml` 只是
  `confirmatory_use_allowed: false` 的单变量设计约束模板；填写 `REQUIRED` 不会自动变成可运行
  pipeline。
- 当前确认性 launcher 只覆盖 breadth/depth；M3/SAGE 的 A1/A3 分析入口不等于
  A1/A3 上游生产链已完成。
- 无论本地测试多完整，都不能替代真实 DeepSeek 调用、ALFWorld 轨迹、四卡
  veRL 训练、vLLM 服务和多 seed 经验结果。
