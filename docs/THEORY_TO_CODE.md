# 理论—实现—验收映射

本文区分三件事：论文中的理想估计量、当前代码实际实现的估计量，以及仍须真实实验
检验的经验主张。任何一项离线测试通过，都不能替代 DeepSeek、ALFWorld、GPU 训练或
vLLM rollout。

## 1. 状态、完整历史与实际模型上下文

论文的完整环境状态为

\[
z_t^{\mathrm{full}}=(Task,H_t,O_t,\mathcal A_t).
\]

`AgentState.full_messages` 保存完整已执行历史，用于 provenance 和 replay；受模型上下文
长度限制，Student 与 Teacher 实际收到的是

\[
c_t^S=(P_S,\tau_\Lambda(z_t^{\mathrm{full}})),\qquad
c_t^T=(P_T,\tau_\Lambda(z_t^{\mathrm{full}})),
\]

其中 \(\tau_\Lambda\) 始终保留 system、task、当前 observation 和 admissible actions，
只从最旧的完整 assistant/user 轮次对开始删除。因而代码实现的是“截断上下文上的策略”，
不能写成模型始终看见未经截断的全部 \(H_t\)。

实现：`AgentState`、`ConversationHistory`、`TaskPreservingTruncator`、`rollout_episode`。

验收：角色顺序必须为 `system,user,(assistant,user)*`；当前状态必须先进入完整历史，再执行
截断；action 后只能记录环境返回的新 observation。

## 2. Student 与 Teacher context

`replace_system()` 只替换 system；同一 state 的非 system 消息逐项保持一致。Teacher
annotation 使用 \(P_T\)，训练 builder 无条件重建为 \(P_S\)，避免 state source 或旧
CTRL prompt 成为 learner-context 混杂。

该约束证明的是两者看见相同的截断 state 内容，不证明 Teacher 与 Student 的 tokenizer、
模型能力或采样随机性相同。

## 3. 黑盒 sampled CE 实际匹配什么分布

若所有 Teacher 输出都能被无损解析并执行，则固定 state 上有

\[
\mathbb E_{A\sim q_T}[-\log p_\theta(A\mid s)]
=H(q_T)+D_{\mathrm{KL}}(q_T\Vert p_\theta).
\]

当前代码还执行 parser 与 admissible-action gate。令 \(V_T=1\) 表示一次 Teacher 输出被
解析为可执行动作，则真正进入训练的 action 分布是

\[
q_T^V(a\mid s)=P(A=a\mid s,V_T=1),
\]

所以 sampled CE 对应

\[
\mathbb E[-\log p_\theta(A\mid s)\mid V_T=1]
=H(q_T^V)+D_{\mathrm{KL}}(q_T^V\Vert p_\theta),
\]

而不是未经筛选的原始 \(q_T\)。只有当 validity 概率为 1，或另有对 invalid 输出的完整
建模时，二者才可等同。

当每个 state 请求 \(N\) 次、单次有效率为 \(p_v(s)\) 且条件 iid 时，该 state 至少有一个
训练 target 的概率为

\[
P(K_s>0\mid s)=1-(1-p_v(s))^N.
\]

因此改变 \(N\) 不只改变 Teacher Monte Carlo 方差，也可能通过 retention 改变训练 state
分布。代码保留全部 attempt，并同时报告 sample acceptance、state acceptance 与完全丢失
的 games；不能把过滤后的结果解释为原始 \(q_T\) 的无偏观测。

## 4. Teacher 预算的精确定义

配置实现

\[
B=MN,
\]

其中 \(M\) 是 distinct selected states，\(N\) 是每个 state 的 annotation attempts，
\(B\) 是 **annotation API attempt 数**。API error、空输出、解析失败和 inadmissible action
各占一次预算，SDK 免费 retry 固定为 0，并拒绝只完成一部分 state 的运行。

该 \(B\) 不是 token、美元、延迟或能耗预算；不同 thinking/profile 的调用也未必等成本。
Teacher 用于生成 Teacher-state pool 的 rollout 调用必须另列，不能暗中并入或排除。

## 5. Breadth–Depth：理想 iid 式与真实分层设计

论文中的理想 iid 动机为

\[
\operatorname{Var}(\hat G_{M,N})=
\frac1M\operatorname{Var}_S[\mu(S)]+
\frac1{MN}\mathbb E_S[\Sigma_T(S)].
\]

规范实验并不是从所有 state iid 抽 \(M\) 个点，而是固定 \(G=50\) 个 games，并在每个
game 的 \(T_g\) 个 pool states 中以 SRSWOR 选择 \(m\in\{1,3\}\) 个 turn，故
\(M=Gm\)、\(B=GmN\)。条件于冻结 game list，在 Teacher draws 条件 iid、有限方差且
暂不考虑 validity filtering 时，更贴近实现的式子是

\[
\operatorname{Var}(\hat G\mid\{g\})=
\frac1{G^2}\sum_{g=1}^G
\left[
\left(1-\frac m{T_g}\right)\frac{S^2_{\mu,g}}m
+\frac{\bar\Sigma_{T,g}}{mN}
\right].
\]

若还把 games 看作从某个 super-population 独立抽样，则近似再包含

\[
\frac1G\operatorname{Var}_g(\bar\mu_g)
\]

的 between-game 项；从有限 game 集无放回抽样时还应有相应 finite-population correction。

固定 \(G\) 与 \(B=GmN\) 后，增加 \(m\) 主要降低 within-game state-sampling 项；Teacher
项近似由 \(B\) 决定，between-game 项不会因 \(m\) 增加而消失。因此“breadth 更好”仍只是
有条件的理论预测。实际训练还有上节所述的 \(N\)-dependent acceptance，必须用新实验
检验，不能由该方差式直接宣布结论。

实现：`uniform_per_game_nested_v1` 为每个 game 生成共同 hash-priority 顺序，保证
`depth_50x3` 的50个 states 嵌套于 `breadth_150x1` 的150个 states，并由
annotation manifest 验证真实 \(B\)。在花费任何 Teacher 调用前，标注入口还会从
冻结 pool 与 selection rows 复算每个 game 的 \(m\) 和 selected set。uniform 分支按
预注册 seed 重建精确 hash-priority SRSWOR 子集并核对 \(\pi_{gt}=m/T_g\)；top-score
分支绑定覆盖完整 pool 的 action-level score table，从 admissible-action log scores
重算 entropy，再重建确定性 top-k，并明确其 population inclusion probability 不可识别。

## 6. Teacher-valid 条件下的 game-balanced objective

设 \(\mathcal G_R\) 为至少保留一个有效 state 的 games，\(\mathcal S_g^R\) 为 game \(g\)
中至少有一个有效 Teacher action 的 states，\(K_s\) 为 state \(s\) 的有效 action 数。
当前 `game_state_mean` 对应

\[
\mathcal L_{GB,V}=\frac1{|\mathcal G_R|}
\sum_{g\in\mathcal G_R}\frac1{|\mathcal S_g^R|}
\sum_{s\in\mathcal S_g^R}\frac1{K_s}
\sum_{j=1}^{K_s}\frac1{L_{sj}}
\sum_k\ell_{sjk}.
\]

每个保留 game 总权重为 1、每个保留 state 等权、state 内有效 Teacher samples 等权、
每个 action 内 token 等权。它不是过滤前
\(G_{GB}=G^{-1}\sum_gT_g^{-1}\sum_tg(s_{gt})\) 的无偏恢复，也不包含完全丢失 games。

用分布语言概括：

\[
q_{train}(s)\propto q_{sel}(s)P(K_s>0\mid s),
\]

而 \(P(K_s>0\mid s)\) 在 iid 假设下为上一节给出的 \(1-(1-p_v(s))^N\)。
这是在给定 `q_sel`/选择权重时对 retention tilt 的简写；最终
`game_state_mean` 还会在保留 games、保留 states 和有效 draws 三层分别归一化，
不能把此比例式当成完整的 row-sampling 概率。

## 7. Action-only token CE

论文目标为

\[
\mathcal L_i=-\frac1{L_i}\sum_k
\log p_\theta(a_{i,k}^T\mid s_i,a_{i,<k}^T).
\]

`encode_final_assistant_content()` 对完整对话只调用一次 chat template；generation prompt
必须是完整对话 token 的严格前缀，否则 fail closed。训练、M1、M3 共用同一
`input_ids + target_token_mask`，只监督最终 `Action: <command>` 的内容 token；assistant
header、generation-only thinking bridge、历史 assistant action 与轮次 terminator 均不监督。

若 mask 标记 `input_ids[:,t]`，causal label 对齐必须使用 `mask[:,1:]` 配合
`labels=input_ids[:,1:]`。旧实现的 `mask[:,:-1]` 是 off-by-one。

## 8. N-sample、state weight 与分布式训练

state/action 权重与 token mask 分离。在 uniform-row sampler 下，设数据有 \(D\) 行、目标
权重总和为 \(W\)，某 optimizer step 跨卡实际抽到 \(r\) 个非 padding 行；共享归一量为
\(rW/D\)，并在拆 microbatch 前确定。每个 microbatch 只贡献 weighted numerator，FSDP
梯度平均由显式 data-parallel factor 抵消，padding row 权重为 0。

这使随机梯度对完整加权数据目标无偏，并避免 batch size 1 时 `1/K_s` 或 game weight 被
随机 in-batch self-normalization 抵消。它不表示每一步的数值 loss 等于完整数据 loss；
无偏性是对 uniform sampling/random permutation 而言。

## 9. Correction utility 与机制分析的可解释范围

论文一阶量

\[
U(s)=G_*^\top g_T(s)
\]

要求真实目标梯度 \(G_*=\nabla L_*\)。当前机制入口没有从 task success 构造这个梯度：

- M1 使用预冻结、game-disjoint probe 的 Teacher-CE gradient 代替 \(G_*\)，主报 dot、norm、
  cosine，名称固定为 **held-out surrogate-gradient alignment**。两组 action table
  必须精确复现同一 annotation-pair 的两个 members，先限制到共同 retained-game
  support，且同 state 多 draw 先平均 gradient 再计算非线性指标。game-disjoint
  基于 annotation manifest 中 validity 前的完整 selected games，而不是过滤后仍有
  有效 action 的 rows。任一梯度范数为0时 cosine 数学上未定义，代码不得用
  数值夹断把它报告为0；
- M2 只诊断 \(q_{train}\) 相对同一 common retained-game support 上 game-balanced pool 的
  marginal JS divergence；各组还必须共享精确 Teacher contract，且共同 support 至少
  包含2个 games，才可报 game-cluster interval；
- M3 使用共享 base、A1/A3 两个完成的 \(N=1\) checkpoint 与 A1/A3 两个冻结
  panel 的完整 checkpoint×panel 2×2 交叉计分，估计 whole-checkpoint
  action-imitation transfer。panel sources 必须是 checkpoints 真实使用的 train split，
  validation games 已排除；updated cells 同时绑定 launch 和 completion manifest。
  当前仓库尚无生成这对 A1/A3 checkpoints 的规范 state-selection pair launcher，
  因此这是分析器契约，不是已完成的确认性上游链；
- SAGE 区分 disagreement \(D\)、intervention label \(I\)、Teacher validity \(V_T\)，主报
  executable turns（Student-valid）上的指标；在 inclusion probability 可识别时报告
  game-balanced HT conditionals，并在共同game support上给出A3−A1 Strong差及配对
  game-cluster区间；
- Position 估计单步 \(C_t=Y_t(a_T)-Y_t(a_S)\)，主报
  \(E[C\mid D=1,V_T=1]\)，unconditional 只作 secondary；它不是参数更新后的 \(\Delta J\)。

因此必须保持

\[
\text{uncertainty}\ne\text{disagreement}\ne
\text{intervention necessity}\ne\text{training utility}.
\]

## 10. Counterfactual consequentiality

`run_paired_counterfactual()` 为两种 intervention action 分别创建环境，以同一 environment
seed replay 同一完整 prefix，通过 task/observation/ordered-admissibles fidelity gate，再由
同一个冻结 Student policy 继续。continuation 从 `full_messages` 恢复，并强制改回 \(P_S\)，
而不是从已截断 query 永久丢失更早轮次。禁止读取历史 `won` 充当 Student branch。

总体 game-balanced HT estimate 只有在每个目标 game 均有 sampled states、所有 inclusion
probability 已知且每个 selected state 都有可 replay 的有效 Teacher action时才输出；否则
只输出 observed-sample 描述量并列出不可识别原因。game bootstrap 条件于已经实现的 state
selection，不传播 state-design variance。

## 11. 训练、服务、环境与推断身份

Student-state on-policy身份沿全链传播：state-pool manifest绑定生成轨迹的完整behavior
Student与Tokenizer，annotation pair绑定该pool，训练launcher再要求训练base/Tokenizer与其
内容身份一致。当前确认性launcher只接受一个可内容寻址的完整Student模型目录；若未来支持
base+adapter行为策略，必须先扩展为有序组合身份，不能只传一个别名。

训练成功不再由目录名推断：只有 trainer 与日志管道成功退出、final step checkpoint、
resolved config 和日志均存在，且 LoRA config、唯一 safetensors 权重、A/B tensor 对、
rank/alpha/dtype/shape/offset/target-module 契约通过时，才创建 completion manifest；评测与
M3还从当前目录重算该契约。评测另外要求本地 vLLM wrapper
证明实际 argv、存活 PID、端口、served aliases、vLLM 版本、base/LoRA 内容指纹和 completion
hash；这是本地内容寻址证明，不是针对拥有主机写权限攻击者的密码学签名。

冻结 game list 同时绑定逐 game/trial-directory 字节、ALFWorld/TextWorld/Python 等运行时
版本，以及 master seed 到 per-game TextWorld seed 的确定性映射。评测按 training seed、
Student decoding seed、environment seed 与 bootstrap seed 分字段记录。

M3的六个checkpoint×panel cells不仅彼此使用相同Tokenizer；2个BASE cells与4个updated
cells还必须回连到训练launch中记录的同一Tokenizer内容身份，否则即使join完整，也不是训练
action-token CE上的概率变化。

`paired_hierarchical_bootstrap()` 在每个 model/game 内保持 treatment-control 配对，并可同时
重采 training seeds 与共享 game clusters。只重采 games 的结果只能解释为 conditional on
these checkpoints；bootstrap 本身也不能修复训练/选择阶段未传播的不确定性。

## 12. 可证范围

自动化测试覆盖状态机、预算、selection、token mask、weighted loss、分布式 padding、
manifest fail-closed、M1/M2/M3/SAGE joins、Position symmetry 与统计函数。这支持“当前代码
满足已写明的离线契约”。在没有真实 Teacher、ALFWorld 数据、4-GPU veRL 0.4.1 训练、
本地 vLLM 服务和完整多 seed 重跑时，不能证明成功率、breadth 优势、selection 排序、
state-source 排序或任何论文数值已经复现。此外，state-source YAML 是明确
不可运行的设计约束模板，M3/SAGE 所需 A1/A3 的规范上游生产链也尚未实现；
“分析入口存在”不等于“确认性实验链完成”。
