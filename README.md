# Agent OmniOPD

这是依据论文定义重新整理的、协议优先的 Black-box Agent On-Policy Distillation
代码库。它不把历史实验产物自动视为可信输入，而是显式记录

\[
B=M\times N,
\qquad
q_{\mathrm{train}}(s)\propto
q_{\mathrm{sel}}(s)P(V_T=1\mid s),
\]

并让 rollout、Teacher annotation、action-only SFT、机制分析和评测共享同一套
state、chat-template 与 loss 定义。

## 当前状态

- `src/omniopd/`：修复后的规范实现；
- `scripts/`：采集、构建数据、评测、统计和本地测试入口；
- `configs/`：显式实验协议；
- `tests/`：不依赖真实 API/ALFWorld 的协议回归测试；
- `legacy/`：从两份 Word 提取的原始代码快照，仅用于追溯，不应运行；
- `docs/AUDIT.md`：问题、影响、修复和旧结果处置；
- `docs/FILE_AUDIT.md`：旧代码问题的逐文件索引；
- `docs/MODIFICATION_PLAN.md`：修改顺序、实施状态与验收门槛；
- `docs/THEORY_TO_CODE.md`：公式到实现与测试的映射；
- `docs/REPRODUCTION.md`：从state pool到统计推断的完整重跑顺序；
- `docs/VERIFICATION_REPORT.md`：代码正确性与公式契合性的最终核验；
- `docs/CODE_INVENTORY.json`：交付文件、字节数与SHA256清单。

旧数据由存在上下文错位、Teacher prompt 未使用、loss mask 错位等问题的代码
生成，因此不能通过“只换训练器”修复。确认性结果必须从新的 state pool 开始重跑。

## 最小验证

建议先安装为可编辑包：

```bash
python -m pip install -e '.[test]'
```

```bash
python scripts/run_tests.py
python -m compileall -q src scripts tests integrations
python -m ruff check src scripts tests integrations
```

若安装开发依赖，也可运行：

```bash
pytest -q
```

## 核心工作流

1. 使用 `scripts/collect_state_pool.py` 只采一次 immutable behavior-policy state pool；
2. 使用 `scripts/score_entropy.py`（可选）和 `scripts/select_states.py` 从同一 pool
   产生各实验组的 selection manifest，禁止每组各自重新 rollout；
3. 使用 `scripts/annotate_states.py` 从 resolved experiment config 读取 \(M,N,B\)，
   请求 Teacher，并把 invalid/API-error attempt 一并计入预算；
4. 使用 `omniopd-build-data` 按 game-safe split 构造 Student-context、action-only 数据；
5. 先用 `scripts/validate_experiment_pair.py` 验收 fixed-budget/control 对照，再按
   `docs/VERL_INTEGRATION.md` 启动固定 optimizer-step 的训练；
6. 使用 `scripts/evaluate.py` 在冻结 game list 上配对评测；训练 seed 与统一的
   rollout seed 是两个独立字段，且checkpoint必须绑定训练manifest中的最终step；
7. 使用 `scripts/aggregate_evaluations.py` 生成带 manifest 的 seed×game 数据，再用
   `omniopd-bootstrap` 校验两组 manifest 后同时传播 game 与 training-seed 不确定性。

`scripts/collect_opd.py` 保留为一次性小规模诊断入口；正式 A1/A3/breadth-depth
实验应使用上述分阶段流程，才能证明不同选择策略共享同一行为轨迹总体。

所有真实运行都应把 resolved config、代码 revision、模型/Tokenizer hash、输入输出
SHA256 和 API call ledger 写入 manifest。不要依赖文件名表达实验协议。
规范入口要求仓库已有Git base revision，并逐阶段比较精确的canonical code-tree hash；
因此不要在一个实验链条中途修改代码或README。

机制分析同样执行来源绑定：M1 action table 必须带规范 build audit；M3 updated score
必须绑定生成最终 adapter 的 training manifest，且 final step、base、Tokenizer与dtype均需
一致。Student uncertainty 只能由生成对应state pool的同一Student/Tokenizer计算。

Teacher 采样必须显式选择 `configs/teacher_sampling_nonthinking.yaml` 或
`configs/teacher_sampling_thinking.yaml`。non-thinking 的 \(N>1\) 使用正温度；thinking
模式省略 temperature 并固定 reasoning effort。两种 profile 不得在比较组间混用。

完整命令、阶段输入输出和重跑边界见 `docs/REPRODUCTION.md`；问题分级、逐文件
处置和公式映射分别见 `docs/AUDIT.md`、`docs/FILE_AUDIT.md` 与
`docs/THEORY_TO_CODE.md`。

## 重要定义

- Teacher budget 默认指 annotation API attempts，invalid 调用同样计费；
- Teacher rollout 本身是否计入总黑盒预算必须在实验协议中单列；
- `game_state_mean` 是 **Teacher-valid 条件下** 的 game-balanced objective，不能声称
  它消除了 acceptance bias；
- M1 是 surrogate-gradient alignment；M3 是 action-imitation transfer，除非另有
  真实 `ΔJ` 验证，不把二者直接称为最终 task utility。
