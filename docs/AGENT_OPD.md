# Agent OPD 主线（黑盒 Agent 在线蒸馏）

本文件是**唯一权威**的方法定义：记号、目标函数推导、实现步骤、四条臂的映射、口径与验收。
其他文档只作为证据或历史记录（见文末"文档地图"）。

## 0. 一句话定义

在**学生自己诱导的状态**上，用**只能给出文本动作的教师**（黑盒：读不到 logprob）提供监督，
以**有效教师动作上的加权交叉熵**更新学生；教师查询次数是显式预算；监督按
game → state → 样本 → token 四级归一。

## 1. 记号与数据流

| 记号 | 含义 |
|---|---|
| $s$ | 状态：$(Task, H, O, \mathcal A)$，即任务、已执行历史、当前观测、当前可行动作集 |
| $\mathcal A(s)$ | 该状态的 admissible actions（环境给定） |
| $P_S, P_T$ | 学生与教师各自的 system prompt（用各自 tokenizer/模板渲染同一段对话） |
| $\pi_\theta$ | 学生策略（LoRA 适配的 Qwen3-4B-Instruct-2507） |
| $\nu$ | 教师策略（Qwen3-14B，**只被查询、只返回文本**） |
| $a_T \sim \nu(\cdot\mid s)$ | 教师在该状态给出的动作文本 |
| $V_T=\mathbb 1[a_T \in \mathcal A(s)]$ | 教师动作是否可执行（有效性） |
| $M, N, B=MN$ | 被选状态数、每状态教师采样次数、教师尝试预算 |

数据流：

```
状态池 ──选择(选样策略/随机)──► 被选状态 M
      └─► 教师查询（黑盒，文本，N 次/状态）──► 有效性筛选 V_T
                └─► SFT 行（state_weight, action-only mask）
                          └─► 加权交叉熵 ──► 梯度下降 ──► 定期评测
```

## 2. 理论推导

### 2.1 一般 OPD 目标

在线蒸馏（on-policy distillation）在**学生诱导的历史** $h_t$ 上让学生匹配教师：

$$
\mathcal L_{\mathrm{OPD}}(\theta)=\mathbb E_{\tau\sim\pi_\theta}\Big[\sum_{t=0}^{T-1}
D\big(\pi_\theta(\cdot\mid h_t)\,\big\|\,\nu(\cdot\mid h_t)\big)\Big]
$$

$D$ 的取法决定需要什么信号：

| $D$ | 方向 | 需要教师提供 | 可否黑盒 |
|---|---|---|---|
| $\mathrm{KL}(\nu\|\pi_\theta)$ | forward | $\log\nu(a)$ 对**采样自 $\nu$** 的 $a$ | **可以**（只需 $\nu$ 的样本） |
| $\mathrm{KL}(\pi_\theta\|\nu)$ | reverse | $\log\nu(a)$ 对**任意** $a$（含学生采样） | **不可以**（需要 logprob 接口） |

**这是"黑盒"的全部含义**：只能拿到 $\nu$ 的**样本**，因此可估的只有 forward 方向。

### 2.2 黑盒情形下的目标（本方法）

forward KL 展开：

$$
\mathrm{KL}\big(\nu(\cdot\mid s)\,\big\|\,\pi_\theta(\cdot\mid s)\big)
=\underbrace{\mathbb E_{a\sim\nu}\big[\log \nu(a\mid s)\big]}_{\text{与 }\theta\text{ 无关}}
-\mathbb E_{a\sim\nu}\big[\log \pi_\theta(a\mid s)\big]
$$

第一项不含参数，于是

$$
\min_\theta \mathrm{KL}(\nu\|\pi_\theta)\iff
\max_\theta \mathbb E_{a\sim\nu}\big[\log\pi_\theta(a\mid s)\big]
$$

用一次教师采样做蒙特卡洛估计，得到**黑盒目标**：

$$
\hat\ell(s)=-\frac{1}{\sum_k m_k}\sum_k m_k \log \pi_\theta\!\left(a_{T,k}\mid s,a_{T,<k}\right),
\qquad a_T\sim\nu(\cdot\mid s)
$$

其期望可写成"熵 + KL"：

$$
\mathbb E\big[\hat\ell(s)\big]=H(q_T)+D_{\mathrm{KL}}\big(q_T\,\|\,\pi_\theta\big)
$$

**要点**：$a_T$ 采自 $\nu$，与 $\theta$ 无关，因此

$$
\nabla_\theta \hat\ell=-\frac{1}{\sum_k m_k}\sum_k m_k \nabla_\theta\log\pi_\theta(a_{T,k}\mid s,a_{T,<k})
$$

**不需要 REINFORCE 修正项**。这正是黑盒 forward-CE 比"在学生采样 token 上估计 reverse KL"方差更低、方向更稳的原因。

### 2.3 有效性筛选

实际进入训练的是**条件分布** $q_T^V(a\mid s)=P(A=a\mid s,V_T=1)$，因此

$$
\mathbb E\big[\hat\ell(s)\mid V_T=1\big]=H(q_T^V)+D_{\mathrm{KL}}\big(q_T^V\,\|\,\pi_\theta\big)
$$

**推论（必须在论文里写明）**：被训练的目标**条件于教师有效**，不是未筛选 $q_T$ 的无偏观测。
每状态采 $N$ 次、单次有效率 $p_v(s)$、条件独立时，该状态**至少有一个**训练目标的概率为

$$
P(K_s>0\mid s)=1-\big(1-p_v(s)\big)^N,\qquad K_s=\#\{\text{有效样本}\}
$$

所以要同时报告 sample acceptance、state acceptance 和"整局丢失"的比例。

### 2.4 归一化与预算（估计量的实际形态）

$$
\mathcal L_{GB,V}(\theta)=
\frac{1}{|\mathcal G_R|}\sum_{g\in\mathcal G_R}
\frac{1}{|\mathcal S_g^R|}\sum_{s\in\mathcal S_g^R}
\frac{1}{K_s}\sum_{j=1}^{K_s}
\frac{1}{L_{sj}}\sum_{k}\ell_{sjk}
$$

四层权重各有理由：**game 各占 1**（否则状态多的游戏支配目标）、**state 等权**、**状态内有效样本等权**（否则多采就多加权）、**token 均权**。
预算 $B=MN$（$M=Gm$：$G$ 个 game、每 game $m$ 个状态），每次 API 尝试——包括无效与解析失败——都计入。

### 2.5 与参考论文（SAGE-OPD, arXiv 2606.19659）的关系

| | SAGE-OPD | 本方法（黑盒 Agent OPD） |
|---|---|---|
| 目标 | Eq.(5) reverse KL $D_{\mathrm{KL}}(\pi_\theta\|\nu)$，**需要教师 logprob** | forward CE，**只需教师文本** |
| 状态/轨迹 | 学生诱导的**多轮轨迹** | 学生诱导状态（先单步池，后可扩到轨迹） |
| 选择性 | 每 turn 的 $z_t\in\{\textsc{Skip},\textsc{Weak},\textsc{Strong}\}$ → $s_t=i_tc_t$ 乘在该 turn 的 loss 上 | **状态/预算选择**（选哪些状态花 $B$） |
| 无效处理 | 无效/不可解析 ⇒ $i_t=1$（**加强**监督） | 记录并**条件于有效**（不训练无效样本） |
| 额外成本 | 每 turn 一次 intervention query（用 two-rollout 基线控制） | 每次 teacher attempt（$B=MN$） |

**两者不可互换**：把"教师动作 CE"称作"传统 OPD"会与论文的分类冲突（论文把"用教师轨迹做模仿"定义为
off-policy multi-turn SFT 基线）。四条臂的对应关系见 §4。

## 3. 实现（每一步映射到现有代码）

| 步 | 做什么 | 现有实现 | 产物 |
|---|---|---|---|
| 1 | 采集学生诱导状态池（含完整历史、admissible、种子） | `scripts/collect_state_pool.py`、`src/omniopd/schema.py` | `state_pool.jsonl` + manifest |
| 2 | 冻结 game 划分与选择（$G$、$m$、嵌套 breadth/depth） | `scripts/freeze_game_list.py`、`src/omniopd/selection.py`、`sampling.py` | selection rows + manifest |
| 3 | **教师黑盒查询**（文本、温度 0、每状态 $N$ 次、记 ledger） | `scripts/annotate_states.py`（需接 vLLM 生成；离线原型见 `scripts/pilot_sft_teacher_rollouts*.py`） | `CorrectionRecord`（state + teacher_samples + validity） |
| 4 | 组训练行（加权 + action-only mask） | `src/omniopd/dataset.py: build_training_rows(weighting="game_state_mean")`、`tokenization.encode_final_assistant_example` | SFT 行（`state_weight`, `target_token_mask`） |
| 5 | 训练（四级归一加权 CE） | `src/omniopd/loss.py: weighted_causal_ce_components` + veRL SFT trainer | LoRA checkpoint + 日志 |
| 6 | 评测（同批配对） | `scripts/host/opd_eval_chain.sh`、`scripts/evaluate_coldstart_sft.py` | 140 局配对 JSON |

实现约定（已验证过的坑）：

* mask **必须** `target_token_mask[:,1:]` 配 `labels=input_ids[:,1:]`（旧 `[:,:-1]` 是 off-by-one）；
* 只监督 `Action: <command>` 的**内容 token**：assistant header、thinking bridge、历史动作、轮次终止符都不监督；
* 教师用**自己的 tokenizer/模板**渲染同一段对话（跨模板模型对必须如此，实测否则首 token 目标会变成 `<think>`、差 21 nats）；
* 训练/评测的 prompt 必须是同一份冻结文本；
* LoRA 的 LR 必须按 `‖ΔW‖/‖W‖` 标定（门槛 ≥1e-3），不能沿用全参数微调的值。

## 4. 四条臂

| 臂 | 状态选择 | 监督信号 | 需要的教师能力 |
|---|---|---|---|
| A0 无更新基线 | — | — | — |
| A1 传统逐 Token OPD | 全部状态（均匀） | 学生自身 token 上的 reverse-KL 估计（PG OPD） | **logprob（白盒）** |
| A2 随机状态-动作校正 | **随机**选 $M$ 个状态 | 教师有效动作的 forward CE | 只要文本（黑盒） |
| A3 所提选样 | **策略**选 $M$ 个状态 | 同上 | 只要文本（黑盒） |

A2 与 A3 只差 $w(s)$（选样权重），因此 A3−A2 是干净的"选择效应"；A1 与 A2/A3 的差别是"监督单位 + 信号类型"，
必须在论文里分开声明，不能混为一谈。

## 5. 冻结项与口径（先定死，再跑）

1. **预算**：黑盒臂 $B=MN$（attempt 计数，含无效与失败）；白盒臂按 teacher token 打分数另行记账。
2. **无效策略**：记录 + 计入预算 + **不训练**（与 SAGE-OPD 的"无效⇒加强监督"不同，须在论文中说明差异）。
3. **评测**：ALFWorld valid_seen 140 局（+ 未来 valid_unseen 134 局），30 轮上限、贪心、AgentBoard 冻结提示词；
   对比一律用**同批配对**（同 game id、同环境种子），报告 McNemar 与分任务族明细。
4. **种子**：game 冻结列表 + 每 game 环境种子由 master seed 确定性派生。
5. **报告**：成功率、无效动作率、总步数、acceptance（sample/state/game 三层）、预算实际消耗。

## 6. 已验证的事实（证据）

* **A1 臂已跑通并显著超过基线**：`opd_native_20260924_01`（4 卡、LoRA r16、LR 3e-5、86 步、`ΔW/‖W‖=1.23e-3`）
  → 同批 140 局 **66/140 = 47.14% vs base 50/140 = 35.71%，净 +16 局，McNemar p = 0.0328**（见 `RESULTS_native_opd_20260924.md`）。
* 反例（同一 harness 的旧实现）：教师用学生模板渲染 + `k2` + LR 1e-6 → 49/140、配对 −2、p=0.75。
  **三处改动（教师渲染、目标函数、LR 标定）缺一不可**。
* 工程侧已验证：docker 环境与镜像身份、checkpoint→PEFT→服务→评测链、ΔW 门槛、CI lint 契约。

## 7. 多轮验证清单（每轮都要过）

| 轮次 | 检查 | 通过标准 |
|---|---|---|
| 公式轮 | 目标函数与论文/推导一致；四层归一每层有理由；无效与预算口径写明 | 本文 §2–§5 无自相矛盾 |
| 代码轮 | mask 对齐、权重公式、token 化与模板、断言 | `tests/`（含 `test_loss.py`、`test_dataset.py`、`test_opd_native.py`）+ `ruff` 全绿 |
| 数值轮 | 加权 CE 与朴素 CE 的差异、权重和归一（§8 分布式归一） | 小样本手算对拍一致 |
| 实跑轮 | 短跑：loss 下降、`ΔW∈[1e-3,1e-2]`、checkpoint 可加载 | 三门槛通过 |
| 评测轮 | 同批配对 + McNemar + 分任务族 | 报告 acceptance 与预算消耗 |

## 8. 架构（干净视图）

```
src/omniopd/            ← 只放"主线需要"的库代码
  schema.py selection.py sampling.py       数据模型与选样
  dataset.py loss.py tokenization.py       训练行与加权 CE
  adapters.py context.py history.py        环境与上下文
  evaluation.py statistics.py provenance.py 评测与统计
  opd_native.py opd_native_layout.py       veRL 原生 OPD + 教师模板钩子（A1 臂）
  # 待实现：teacher_blackbox.py（A2/A3 的教师文本查询与 ledger）
scripts/                ← 可执行入口（采集/训练/评测），一次性实验脚本不放这里
tests/                  ← 离线契约测试（196 项）
docs/AGENT_OPD.md        ← 本文件（唯一权威）
docs/THEORY_TO_CODE.md   ← 公式↔代码逐条映射（本文件的附录性质）
docs/EXPERIMENT_ISSUES_*.md ← 问题台账（历史证据）
docs/RESULTS_*.md        ← 结果记录
```

**已删除的历史文档**（内容在 git 历史里，需要时 `git show <commit>:<path>` 取回）：
`00_START_HERE.md`、`AGENT_R1_ADAPTATION_20260918.md`、`ALFWORLD_REBUILD_20260916.md`、`AUDIT.md`、
`EXECUTION_INPUTS.template.yaml`、`EXECUTOR_PROMPT.md`、`FILE_AUDIT.md`、
`LOCAL_EXECUTION_AND_GITHUB_REQUIREMENTS_ZH.md`、`MODIFICATION_PLAN.md`、
`PLAN_verl_native_opd_20260923.md`、`VERIFICATION_REPORT.md`、`VERL_INTEGRATION.md`、
`VERL_V080_MIGRATION.md`、`ANALYSIS_standard_opd_20260923.md`。
删除理由：都是过程性/被取代/一次性审计文档，主线已在本文与台账中固化。
