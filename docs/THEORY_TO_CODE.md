# 理论—实现—验收映射

## 1. 状态定义

论文：

\[
z_t=(Task,H_t,O_t,\mathcal A_t).
\]

实现：`AgentState`、`ConversationHistory`、`rollout_episode`。

约束：模型查询时角色严格为 `system,user,(assistant,user)*`；当前 observation 与
admissible actions 必须已经进入 history 后才允许截断；action 后只能记录环境返回的
新 observation。

测试：`test_history.py`、`test_protocol.py`。

## 2. Student 与 Teacher context

论文：

\[
c_t^S=(P_S,z_t),\qquad c_t^T=(P_T,z_t).
\]

实现：`prompts.replace_system()` 仅替换 system，非 system 消息逐项保持相同；训练
builder 无条件重建为 `P_S`，避免 CTRL prompt 成为混杂。

测试：Teacher/Student 非 system 消息完全一致，system 分别为 `P_T/P_S`。

## 3. 固定 Teacher 预算

论文：

\[
B=MN.
\]

实现：`TeacherBudget` 对每次真实 API request 计数，包括 invalid；
`validate_budget()` 区分 distinct states、samples/state、actual calls 与 accepted targets。

固定预算 breadth-depth 必须显式比较，例如 `150×1` 与 `50×3`；原实验的
`150×1` 与 `150×3` 不是 fixed-B。

理论预测：

\[
\operatorname{Var}(\hat G_{M,N})=
\frac1M\operatorname{Var}_S[\mu(S)]+
\frac1{MN}\mathbb E_S[\Sigma_T(S)].
\]

实现只保证两个arm共享pool、nested uniform design、Teacher profile、(B)与optimizer
steps；“breadth更好”仍需新实验检验，不能由方差式直接宣布。

## 4. Game-balanced objective 与 acceptance

论文目标：

\[
G_{GB}=\frac1G\sum_g\frac1{T_g}\sum_t g(s_{gt}).
\]

Teacher filtering 后：

\[
q_{train}(s)\propto q_{sel}(s)P(V_T=1\mid s).
\]

实现：`build_training_rows(..., weighting="game_state_mean")` 令每个保留 game 的
总 state weight 为 1，每 state 内 K 个 action 权重和为该 state 权重。该模式明确命名
为 acceptance-conditioned game balance，不声称恢复过滤前分布。

## 5. Action-only token CE

论文：

\[
\mathcal L_i=-\frac1{L_i}\sum_k\log p_\theta(a_{i,k}^T\mid s_i,a_{i,<k}^T).
\]

实现：`encode_final_assistant_content()` 使用一次完整 chat template；若 generation
prompt 不是完整对话 token 的严格前缀则 fail closed。训练、M1、M3复用同一
`input_ids + target_token_mask`，只监督 `Action: <command>` 内容 token；assistant
header、generation-only thinking bridge 与轮次 terminator 均不属于论文中的动作 token。

next-token 对齐：若 mask 标记 `input_ids[:,t]`，则 loss 使用 `mask[:,1:]` 对齐
`labels=input_ids[:,1:]`。历史实现的 `mask[:,:-1]` 是 off-by-one。

## 6. N-sample state weighting

论文：

\[
\frac1{|\mathcal S|}\sum_s\frac1{K_s}\sum_j
\frac1{L_{sj}}\sum_k\ell_{sjk}.
\]

实现：state/action 权重与 token mask 分离。uniform row sampler 下，若训练集有
`D`行、总目标权重为`W`，本步跨卡真实抽到`r`行，则在拆microbatch前固定归一量
`rW/D`。这使随机梯度对全数据加权目标无偏，也避免batch=1时权重再次被抵消。
每个microbatch只贡献 weighted numerator/shared normalizer，禁止在microbatch内
独立归一，也禁止在backward后再除microbatch数。

## 7. Correction utility

论文：

\[
U(s)=G_*^\top g_T(s).
\]

实现原则：M1 主报 dot product、norm 和 cosine，名称为 surrogate reference
alignment；probe 必须来自预先冻结的 held-out games，而非 selection 补集。

M3 的 checkpoint-level `Δlogp` 只能命名为 action-imitation transfer，不能直接等同
于 task utility。必须 crossed-evaluate A1/A3 checkpoints，并在相同冻结 source/neighbor
集合上比较。

M2只诊断 (q_{train}) 相对game-balanced pool的边际JS divergence；SAGE分别测量
disagreement与盲评的intervention necessity。代码明确保持
`uncertainty ≠ disagreement ≠ intervention necessity ≠ utility`，不把其中任一代理量
代入 (G_*^\top g_T(s))。

## 8. Counterfactual consequentiality

论文：

\[
C_t=Y_t(a_T)-Y_t(a_S).
\]

实现：`run_paired_counterfactual()` 为两种动作分别创建环境、replay 同一 prefix、
通过 observation/admissible gate，并调用同一个冻结 continuation policy。禁止读取
历史 `ep["won"]` 充当 Student 分支。

## 9. 推断

实现：评测把训练checkpoint seed与固定rollout seed分开，并绑定final step、base/LoRA、
训练协议、game-list manifest、环境、prompt、Tokenizer和推理runtime。
`paired_hierarchical_bootstrap()` 在 model/game 内保持配对，同时可重采样training seeds。
只重采样 games 的结果只能解释为“conditional on these checkpoints”。

## 10. 可证范围

自动化测试覆盖状态机、budget、selection、token mask、loss、分布式padding、统计join与
counterfactual symmetry。它能证明实现满足上述离线契约；没有实际Teacher、ALFWorld、GPU
训练和vLLM rollout时，不能证明经验成功率、breadth优势或任何state-source排序。
