# Agent OmniOPD

> 最近一次落地（2026-09-20）：传统逐 Token OPD 的 veRL v0.8.0 训练内核已在
> Qwen3-4B Student / Qwen3-14B Teacher 上跑通真实单步——非零梯度、优化器一阶与
> 二阶动量非零、checkpoint 保存与显式恢复续训均通过。但这仍是非确认性单状态烟测：
> invalid 与预算口径、`state_weight` 消费、正式估计器和多游戏数据都未冻结，
> 不要据此启动确认性实验。共同冷启动 SFT 正在同一分支上建设。
> 详见 `docs/EXPERIMENT_ISSUES_20260917.md`（E10、O23–O25）与
> `docs/VERL_V080_MIGRATION.md`。

这是依据论文定义重新整理的、协议优先的 Black-box Agent On-Policy Distillation
代码库。它不把历史实验产物自动视为可信输入，而是显式记录状态 schema、
\(P_S/P_T\) 分离、角色安全的模板规则、action-token 契约以及

\[
B=M\times N,
\qquad
q_{\mathrm{train}}(s)\propto
q_{\mathrm{sel}}(s)P(K_s>0\mid s),
\]

其中 \(K_s=\sum_{j=1}^{N}\mathbf 1\{V_{T,sj}=1\}\)。若同一 state 的
Teacher draws 在给定 \(s\) 后独立同分布，则
\(P(K_s>0\mid s)=1-(1-p_v(s))^N\)。因此改变 \(N\) 不只改变方差，也会改变
进入训练集的 state 分布；代码不会把它误写成与 \(N\) 无关的
\(P(V_T=1\mid s)\)。invalid attempts 占用 \(B\)，但 action loss 的条件目标是
\(q_T^V(a\mid s)=P(A=a\mid s,V_T=1)\)。模型实际接收的是受 token budget 约束的
截断上下文 \(\tau_\Lambda(z_t)\)；`full_messages` 只用于对称 replay 和 provenance，
不伪装成模型看到的输入。
上述比例式是 game/state 分层归一前的 retention tilt 简写；实际
`game_state_mean` 目标还会对保留 games、states 和 valid draws 分层归一。

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

## 版本管理与持续迭代

- `main` 是主干，只接收通过验证的改动；每个任务在独立分支上推进
  （本任务为 `feat/expert-sft-coldstart`），完成后经 PR 或多个提交合并回 `main`；
- 每次提交只承担一种清晰变更；回归数字、镜像标签与提交哈希三者必须能相互对上；
- `.github/workflows/ci.yml` 在 push 与 PR 上运行离线 CI（编译、离线测试、Ruff），
  覆盖 Python 3.10/3.11/3.12；GPU、vLLM、ALFWorld 与 Teacher API 集成测试留在
  自托管 runner 或人工 smoke gate；
- 服务器同步只走经 `git bundle verify` 校验的已提交引用，随后用 `merge --ff-only`
  快进，并在容器内重跑离线回归后记录通过数；
- 里程碑用 tag 标记（例如 `omniopd-v1.0.0`）；正式实验只能从带 tag 的干净工作区
  启动，运行期间不得修改代码，需要修复时新建提交、新 tag 与新输出目录。

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

1. 使用 `scripts/run_vllm_state_pool.sh` 启动并现场核验完整 Student 服务，只采一次
   immutable behavior-policy state pool；pool manifest 会保存规范化的服务证明及其
   内容摘要，后续 selection、annotation、pair、训练与 Position 均沿链复核；
2. 使用 `scripts/score_entropy.py`（可选）和 `scripts/select_states.py` 从同一 pool
   产生各实验组的 selection manifest，禁止每组各自重新 rollout；标注前会从实体 rows
   复算每局 (m)、状态集合与 SRSWOR 纳入概率 (m/T_g)。若使用 top-score，选择与标注
   两个入口都必须绑定完整 score rows/manifest，并从逐 admissible-action log score
   重新计算 entropy，不能只信任缓存的 state-level score；
3. 使用 `scripts/annotate_states.py` 从 resolved experiment config 读取 \(M,N,B\)，
   请求 Teacher，并把 invalid/API-error attempt 一并计入预算；
4. 使用 `omniopd-build-data` 按 game-safe split 构造 Student-context、action-only 数据；
5. 使用 `scripts/validate_experiment_pair.py --kind annotation_runs --output ...` 生成唯一的
   annotation-pair manifest；两个训练臂必须共同绑定这个完整文件，再按
   `docs/VERL_INTEGRATION.md` 启动固定 optimizer-step 训练；
6. 训练正常退出且 final checkpoint 被验证为超参数匹配、带 Tokenizer、A/B tensor
   成对且覆盖配置 target modules 的 PEFT LoRA adapter 后才生成 completion manifest。
   使用 `scripts/run_vllm_eval.sh` 在服务仍为
   wrapper 子进程时核验 base、最终checkpoint、launch/completion manifest 与运行时，
   然后在冻结 game list 上配对评测；training seed、rollout seed 与 environment seed
   是三个独立字段；
7. 使用 `scripts/aggregate_evaluations.py` 生成带 manifest 的 seed×game 数据，再用
   `omniopd-bootstrap` 校验两组 manifest 后同时传播 game 与 training-seed 不确定性。

`scripts/collect_opd.py` 保留为一次性小规模诊断入口。当前能从头闭环的确认性
流程是 fixed-budget breadth/depth；A1/A3 的分析入口已实现，但其上游规范
state-selection pair/launcher 尚未提供，不能写成可生成确认性 A1/A3 结果。

所有真实运行都应把 resolved config、代码 revision、模型/Tokenizer hash、输入输出
SHA256 和 API call ledger 写入 manifest。开始任何实验链前，先提交本仓库并保持
working tree 干净；canonical code-tree hash 覆盖 `src/`、`scripts/`、`integrations/`、
`configs/`、`pyproject.toml` 和 `README.md`。链条中途修改其中任一文件都会
故意触发身份校验失败；不要依赖文件名表达实验协议。

机制分析同样执行来源绑定：M1 action table 必须带规范 build audit；M2 只在共同
Teacher-valid game support 上比较；M3 使用 checkpoint×panel 的 2×2 交叉设计，updated
score 同时绑定 launch 与 completion manifest，且 final step、base、Tokenizer与dtype均需
一致；SAGE 先构造跨实验组去重并集再盲评。Student uncertainty 只能由生成对应state pool
的同一Student/Tokenizer计算。Position 必须由 `scripts/run_vllm_position.sh` 启动生成
该 state pool 的同一完整 behavior Student/Tokenizer/runtime，而不是训练后 adapter。

Teacher 采样必须显式选择 `configs/teacher_sampling_nonthinking.yaml` 或
`configs/teacher_sampling_thinking.yaml`。non-thinking 的 \(N>1\) 使用正温度；thinking
模式省略 temperature 并固定 reasoning effort。两种 profile 不得在比较组间混用。

完整命令、阶段输入输出和重跑边界见 `docs/REPRODUCTION.md`；问题分级、逐文件
处置和公式映射分别见 `docs/AUDIT.md`、`docs/FILE_AUDIT.md` 与
`docs/THEORY_TO_CODE.md`。

当前发布包的确认性 launcher 完整实现的是 fixed-budget breadth/depth 二臂链。
`student_state_control.yaml`、`teacher_state_control.yaml` 明确只是不可运行的设计约束模板；
M3/SAGE 分析器也要求两个已完成的 (N=1) A1/A3 checkpoint/correction group，
但仓库尚未提供生成这对 state-selection 运行的规范 pair launcher。因此这两类
结果会 fail closed，不能把“分析
函数已实现”写成“确认性实验生产链已完成”。

## 重要定义

- Teacher budget 默认指 annotation API attempts，invalid 调用同样计费；
- Teacher rollout 本身是否计入总黑盒预算必须在实验协议中单列；
- `game_state_mean` 是 **至少一次Teacher-valid的保留state条件下** 的 game-balanced
  objective；由于保留概率随 \(N\) 变化，不能声称它消除了 acceptance bias；
- M1 是 surrogate-gradient alignment；M3 是 action-imitation transfer，除非另有
  真实 `ΔJ` 验证，不把二者直接称为最终 task utility。
