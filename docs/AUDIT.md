# 代码审计与处置记录

## 总结

原 Word 中的 Python/Shell 均可通过基础语法解析，但关键问题属于语义、数学和实验
协议错误。旧结果不得通过简单改名继续作为确认性证据；必须重采核心 state pool、
重建数据、重训并重评。

## P0：必须重跑的原因

1. **State/history 错位**：初始 observation 被 task description 替代；`env.step()` 后
   写入旧 observation；当前 state 在截断后才临时追加，既产生连续 user，也逃逸 token
   budget。影响 A1、A3、SFT、CTRL 和 eval。
2. **Teacher prompt 未生效**：代码定义 `P_T`，请求却直接发送带 `P_S` 的
   `query_messages`。
3. **CTRL 不是单变量对照**：state source、prompt、Thought格式、admissibles、thinking、
   correction协议和 learner context 同时变化；invalid fix 还引入双 system 与覆盖原文件。
4. **训练 mask off-by-one**：target mask 用 `[:,:-1]` 对齐 next-token label，应为
   `[:,1:]`。
5. **梯度累积缩放无效**：loss 已 backward 后才除 microbatch 数；1-GPU/4-GPU更新
   不等价。
6. **A2 state weight 被抵消**：microbatch=1 时，每 microbatch内部按自身权重归一，
   `1/K_s` 完全消失；v2 builder 与 launcher 又断开。
7. **M1/M3 token 不是训练 token**：缺 assistant header，target可能插入 BOS，EOT与
   thinking配置不一致。
8. **M3统计损坏**：A1/A3按 game 混合、交互项裸索引取错、95%上界取成99.5%。
9. **SAGE覆盖损坏**：technical states 在 judge前被删除，DetStrong分支不可达；
   `teacher_actions` dict 被当字符串比较。
10. **Position Study不对称**：只重跑 Teacher branch，Student outcome来自旧轨迹。
11. **主数据 provenance 断裂**：A1R42生成源码不可恢复；脚本与训练日志、dtype和模型
    路径存在不一致。
12. **ALFWorld task family 解析错误**：旧代码从 split/trial 层级切字符串，M2/M3 的
    `task_type` 分层与邻居约束会被系统性污染。
13. **评测 seed 混义**：同一个 `seed=42` 同时被当作 checkpoint 训练身份、环境/解码
    控制和 bootstrap RNG；无法证明多 seed checkpoint 使用相同 rollout regime。
14. **评测与 checkpoint 未绑定**：旧结果只靠文件名表达模型，未绑定 final step、训练
    manifest、base、LoRA、冻结 game list 或推理引擎版本。
15. **机制分析输入可裸接**：旧 entropy/M2/M3/SAGE 脚本可接受来源不明或来自另一 state
    pool 的 JSON；M3 也会把 LoRA adapter 目录当完整 Hugging Face 模型加载。
16. **Position continuation 丢失历史**：即便两分支都 replay，若从已截断 query 而非
   `full_messages` 继续，后续 context 仍与真实 frozen Student policy 不同；Teacher-state
   输入还可能让 continuation 错用 (P_T)。
17. **“清单自述”可替代“实体事实”**：早期重写版只核对部分 manifest 字段和文件哈希，
   没有在 M2、Position 等消费者处从 corrections/traces 重新计算 (M,N,B)、有效/无效数、
   selected-state 集合与逐 game 结果；一个内部不一致但格式合法的清单仍可能通过。
18. **实验对只绑定计划、未完整绑定实现**：若 pair artifact 不绑定两个 realized annotation
   manifest 的全部 Teacher、pool、selection、预算与 invalid-policy 契约，训练时可以替换
   某一臂的数据或颠倒 breadth/depth 方向。
19. **训练完成与在线服务身份可伪装**：仅有 launch manifest 或模型 alias 不能证明训练已
   正常完成，也不能证明评测时实际存活服务加载了相应 final checkpoint。
20. **环境随机性与游戏字节未闭环**：只记录 game 路径或 Python seed 不能证明 TextWorld
   使用同一环境种子、同一 trial 内容和同一 ALFWorld/TextWorld runtime。
21. **行为 Student 与训练/M3 Tokenizer 可漂移**：state pool 可以由模型A产生、再用模型B
   训练；M3也可让六格共同使用一个与训练不同的Tokenizer，形式上一致但不再测量同一个
   on-policy action-token目标。
22. **单簇置信区间与零梯度 cosine 被伪数值化**：一个 game 的所谓cluster bootstrap没有
   可估计的簇间不确定性；零范数梯度的 cosine 未定义，不能通过数值夹断报告为0。
23. **SAGE 主估计量口径漂移**：把 technical/Student-invalid states 放进 (U_g) 分母会把
   预注册的 executable-turn 指标改成全 selected-state 指标；只报各组率也不能替代
   A3−A1 的配对 game-cluster 对比。
24. **跨阶段 behavior Student 身份链曾断开**：规范重写版的 selection 与 correction
   manifest 起初没有写出下游验证器强制要求的 `behavior_student`，导致
   select→annotation 及 annotation→pair 两处必然失败；现已由 state-pool manifest
   统一派生并通过端到端回归测试。
25. **selection 清单可伪造抽样设计**：仅核对 selection 文件哈希与总 (M) 仍允许
   清单虚报逐 game 数量、selected hashes 或 row-level inclusion probability，使 HT/
   Position 使用错误的 (m/T_g)；标注入口现从冻结 pool 与 selection rows 逐项复算。
26. **非空 checkpoint 不等于可加载 LoRA**：训练正常退出且 final 目录非空，仍不能排除
   保存成完整模型、adapter 文件缺失或 rank/alpha 与 launch 不同；trainer 与 completion
   生成器现要求 PEFT LoRA config、唯一 safetensors adapter 权重、无完整模型权重、Tokenizer config、
   匹配超参数、成对 A/B tensors、合法 shape/dtype/offset 与 target-module 覆盖；评测和
   M3 消费时从 checkpoint 实体再次重算，不只信任 completion 自述。
27. **state-pool 服务证明曾在下游丢失**：pool 虽由受控 wrapper 生成，但早期
   `behavior_student` 投影未携带规范化 service attestation，后续只验证模型目录内容，
   无法证明轨迹来自声明的存活本地服务；现将服务 manifest 摘要与路径无关证明沿
   selection→annotation→pair→训练/Position 全链传播并逐字段复核。
28. **top-score 可只信任缓存 entropy**：仅绑定 score 文件哈希仍允许同一文件内把
   state-level score、admissible action 集或模型身份写错，导致 deterministic top-k
   选择偏离预注册定义；现要求 score manifest 绑定同一 behavior Student/Tokenizer/pool，
   score rows 与全部 pool states 一一对应，并从每个 admissible action 的有限 log score
   重算 entropy 后再选择或标注。

## P1：主张需要降级或补实验

- 原 A2 是 `M固定、N增加`，预算扩大三倍，不是 fixed-B breadth-depth；
- valid filtering 与短轨迹使最终训练目标不再是未条件化的 `q_GB`；
- matched-count 只匹配 parquet rows，未匹配 API calls、states、tokens、tasks或steps；
- M1 reference 是 selection-defined补集上的Teacher-CE surrogate，不是成功率梯度；
- M2使用turn-pooled baseline而非 game-balanced baseline，且只比较边际分布；
- M3是whole-checkpoint action likelihood shift，不是单条 correction utility；
- SAGE人工抽样是定向分层抽样，未经 inclusion-probability 加权不能报告总体概率；
- 旧 bootstrap只重采样games，未传播训练seed与subsample随机性；
- seed42与其他seed provenance不可交换。

## 规范实现已经采取的修改

- `AgentState + ConversationHistory + TaskPreservingTruncator` 对
  (z_t=(Task,H_t,O_t,\mathcal A_t)) 做单一状态机管理，并保存完整历史、实际 query 与
  截断证明；当前 observation/admissibles 与序列角色逐项审计。
- Student rollout、Teacher query 与训练分别固定 (P_S/P_T/P_S)，Teacher 只替换 system，
  非 system context 的哈希必须相同；所有 API attempt（包括异常和 invalid）占用预算且
  SDK retry 固定为 0。
- breadth `150×1` 与 depth `50×3` 从同一 pool 的嵌套 SRSWOR selection 产生，严格满足
  (B=MN=150)；pool seed、selection seed、data-split seed、training replicate seed、
  rollout seed 与 bootstrap seed 均为不同字段。
- `game_state_mean` 对每个 Teacher-valid state 等权、每个保留 game 等权；训练 audit 同时
  报告 state/sample acceptance 和完全丢失的 games，不把它写成未条件化的 (q_{GB})。
- action-content-only chat-template mask 被训练、entropy、M1、M3 共用；next-token 对齐为
  `mask[:,1:]`，assistant header、thinking bridge 和 terminator 全部排除。
- uncertainty score 强制使用生成该 Student-state pool 的同一模型与Tokenizer内容指纹；
  score rows 必须覆盖且只覆盖全部 pool states，admissible action 集、game/turn 与 pool
  一致，并从 action log scores 重算 entropy，防止把其他checkpoint或篡改分数的不确定性
  误标为当前Student uncertainty。
- 训练以 uniform-row 抽样的期望质量 `rW/D` 归一，microbatch 只累加 numerator；FSDP
  padding 权重为 0，optimizer step、precision、veRL 版本和四个 training seeds 显式冻结。
- 评测结果必须绑定 final `global_step`、training manifest、base/LoRA 内容指纹、冻结 game
  list manifest、稳定 served-model alias、vLLM 版本和独立 rollout seed；推断同时重采
  配对的 training seed 与 game。
- M3 先冻结跨 game、同 task、action-admissible、未被任何 design 选中的邻居 panel，再用
  共同 base 与两个由规范训练 manifest 绑定的最终 adapter 计分；M1输入也绑定完整未划分
  action-table audit；SAGE blind judge 不接收 Teacher/entropy/group，
  失败标签保持 missing；Position 从完整历史恢复 (P_S) 后对称 replay 两个动作分支。
- annotation-pair 是一个内容寻址的二臂整体：完整绑定 realized manifests、共同 Teacher/
  state-pool/selection 契约、arm role 与 `breadth_minus_depth` 方向；训练 launch 与 completion
  均继承这一身份，不能只凭两个配置文件声称形成有效对照。
- selection 与 correction manifest 均携带由同一 state-pool manifest 规范化得到的完整
  behavior Student 契约；端到端测试覆盖 pool→selection→annotation→pair，防止生成者与
  消费者各自通过、串联却失败。
- annotation 在任何 Teacher 调用前，从 selection rows 与冻结 pool 复算每个 game 的
  (m) 和 selected set；uniform 分支按 seed 重建精确 hash-priority 子集并核对 (m/T_g)，
  top-score 分支按完整 score table 重建确定性 top-k、核对并列顺序且不得伪称 design
  probability。
- FSDP 保存后各 rank 验证产物是匹配 launch rank/alpha/`all-linear` 的 PEFT LoRA
  adapter，并包含唯一 safetensors 权重、Tokenizer配置及成对 A/B tensors；completion
  schema v2记录其内容契约，消费者从当前目录重新解析并与之精确比较。
- 本地 vLLM wrapper 在服务仍存活时记录父子进程、实际 argv、端口、runtime、模型与
  Tokenizer 字节身份；评测、state-pool 与 Position 消费者现场复核。该证明的信任边界是
  同一受信主机，并非抵抗本机管理员的密码学远程证明。
- game list/state pool 记录 trial 目录内容、环境依赖版本和 master-to-per-game seed 映射；
  训练基础模型绑定生成 pool 的完整 behavior Student，M3 Tokenizer绑定训练身份。
- 机制入口不再信任清单自述：在使用前从 records/traces 重算关键计数与结果。M1零范数
  cosine 明确不可识别，M2/Position/SAGE 的 cluster interval 至少需要两个 games；SAGE
  主报 executable-turn (U_g)，并给出共同game support上的A3−A1配对比较。

## 修复后的解释边界

- sampled action CE 可支持 black-box policy distillation 恒等式；
- `game_state_mean` 支持 Teacher-valid 条件下的 game balance；
- M1 只称 surrogate-gradient alignment；
- M3 只称 action-imitation transfer，除非另有真实 `ΔJ` 验证；
- Position Study只估计预注册 state population 中的单步 action consequentiality，不等同
  于完整训练 utility；
- Student-state 与 Teacher-state 的价值应按 evaluation regime alignment 解释，不声称
  某一 source 天生更好。

## 仍需真实实验验证的边界

本地代码核验不能替代 DeepSeek 调用、ALFWorld 轨迹、四卡 veRL 训练或在线 vLLM 评测。
因此本仓库证明的是“协议实现与公式一致并能 fail closed”，不是证明论文的经验结论已经
重新成立。所有确认性数值必须按 `REPRODUCTION.md` 从新 state pool 重跑；CTRL 配置目前是
明确不可运行的单变量设计约束模板，仅填入 `REQUIRED` 不会生成 state-source
结论。当前确认性 launcher 只闭环 fixed-budget breadth/depth；M3/SAGE 虽有严格
A1/A3 分析入口，但尚无两个 \(N=1\) state-selection arms 的规范 pair/training 生产链。

## 旧代码处置

`legacy/main_document` 与 `legacy/supplement_document` 是不可变审计快照。它们用于
定位历史产物，不是 canonical runtime。新实验只允许导入 `src/omniopd`，运行时应记录
模块真实路径与 SHA256，防止 `/cfs`、`/root/code`、`/root/data` 多代码根漂移。
