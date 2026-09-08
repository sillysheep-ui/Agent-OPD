# 修改方案与重跑边界

## 修改顺序

1. **冻结协议**：固定prompt、parser、fallback、thinking、模型revision、游戏列表、
   \(M,N,B\)、训练总步数和评测regime；所有值进入manifest。
2. **重采state pool**：用统一`rollout_episode()`重新生成Student-state ledger；旧A1/A3/SFT
   state不得进入确认性实验。
3. **从同一pool选择**：当前实现的 uniform random 与 top-score（如 entropy）策略
   只输出selection manifest，不各自重新rollout。bounded/mixed 若要进入确认性实验，
   必须先实现可审计的 inclusion probability 与对应加权，不能沿用旧脚本。
4. **统一Teacher annotation**：只替换system为\(P_T\)，每个真实attempt均计入预算，
   invalid保留且不免费重试。
5. **重建action-only数据**：训练始终使用\(P_S,z_t\)，最终assistant target仅含
   `Action: <canonical command>`；按game/state/sample显式加权并按game划分验证集。
6. **重训**：使用patched trainer；uniform row sampling使用`rW/D`固定归一量并在拆
   microbatch前计算；分布式padding样本权重为0；各比较组固定总optimizer steps与dtype。
7. **配对重评**：所有checkpoint复用相同冻结ID/OOD游戏列表与rollout engine。
8. **重做机制分析**：M1/M3复用训练token helper；M2按game balance；SAGE保留全state ledger；
   Position对两个动作各自完整replay。
9. **统计推断**：处理seed与game两个不确定性层级；报告point estimate、95%区间、各seed结果、
   attrition与acceptance。
10. **对规范实现做对抗式复核**：不只构造正常输入，还逐项替换manifest、成员arm、模型、
    Tokenizer、checkpoint、service PID、环境seed和派生统计，确认每个下游消费者从实体重算
    并 fail closed。

## 实施状态

| 阶段 | 状态 | 规范入口/结果 |
|---|---|---|
| 1 协议冻结 | 已完成（代码层） | `configs/`、各阶段manifest、模型/Tokenizer/代码指纹 |
| 2 统一state pool | 已完成（实现） | `collect_state_pool.py`；真实pool需重采 |
| 3 selection | 已完成 | 精确重建nested uniform；top-score绑定动作级score并重算entropy |
| 4 Teacher annotation | 已完成 | 精确`B=MN`、无免费retry、context preflight与request ledger |
| 5 action-only数据 | 已完成 | (P_S) context、action-content mask、game-safe split与acceptance audit |
| 6 固定步数训练 | 已完成（集成代码） | veRL 0.4.1/FSDP trainer；目标四卡环境尚未实际运行 |
| 7 配对评测 | 已完成（入口） | checkpoint/training/game-list/runtime绑定；真实vLLM/ALFWorld尚未运行 |
| 8 机制分析 | 已完成（分析入口） | M1/M2/M3/SAGE/Position均有明确估计量和输入清单；M3/SAGE的A1/A3上游生产链未完成 |
| 9 推断 | 已完成 | paired training-seed × game hierarchical bootstrap |
| 10 第二轮独立复核 | 已完成（离线） | records重算、pair/completion/service/environment身份链、单簇与不可识别性负测 |

“已完成”表示表中明示范围的代码与协议已实现并通过本地核验，不表示
新的论文实验数值已经产生。当前确认性 launcher 只闭环 fixed-budget
breadth/depth；state-source controls 是不可运行模板，A1/A3 state-selection pair
launcher 尚未实现。

## 哪些旧产物可以保留

- 可以保留：原始文档、日志、checkpoint和提取代码，作为历史追溯材料；
- 只能探索性使用：M2/SAGE人工标注中仍能一一回连到完整state的原始标签；
- 必须重算：所有依赖训练chat-template的M1/M3数值；
- 必须重跑：A1/A3/SFT/CTRL state pool、所有由其训练出的确认性checkpoint和评测；
- 不得混合：来源无法恢复的seed42与新协议seed replicates。

## 验收门槛

- 每个query角色严格交替，当前observation/admissibles已包含在截断预算内；
- Teacher与Student非system消息逐项相同，system分别为\(P_T/P_S\)；
- `actual_teacher_api_calls == distinct_states_M * samples_per_state_N == B`；
- invalid attempt总数、接受率和被丢弃game均显式报告；
- 训练、M1、M3对同一样本的`input_ids`与目标mask逐token一致；
- 任意microbatch划分得到相同optimizer-batch loss/gradient；
- breadth/depth组Teacher calls和optimizer steps均相等；
- control除`state_source/behavior_policy`外无协议差异；
- Position中`a_T=a_S`时两支trace相同且\(C_t=0\)；
- 所有数据/模型/代码入口均有hash，重复或覆盖写入默认失败。
- 评测checkpoint必须是training manifest声明的最终`global_step`，base与adapter身份可回溯；
- M3必须完成2个BASE cells与4个updated cells的checkpoint×panel 2×2交叉设计；
  六次计分共享base/Tokenizer/dtype，两个updated adapter必须不同；
- M1每张action table必须绑定规范build audit；M3 updated adapter必须绑定各自训练manifest的final step；
- SAGE缺失judge标签不插补，未知inclusion probability时不得报告总体率。
- annotation pair 必须恰有两个不同实验成员，完整绑定realized annotation bytes、同一state
  pool/Teacher契约、arm roles与breadth-minus-depth方向；两臂训练共同绑定同一pair文件；
- 训练base和Tokenizer必须与生成Student state pool的behavior Student及其服务证明一致；
  训练正常退出后且final目录通过PEFT LoRA语义、唯一safetensors、Tokenizer、rank/alpha、
  成对A/B tensors与target-module覆盖核验后，
  completion manifest才可被评测或M3使用；
- selection 的每局 (m)、selected hashes、game support 与 (m/T_g) 必须在任何Teacher
  调用前从冻结pool和selection rows复算，不接受manifest自述替代；uniform 必须重建
  seeded hash-priority 子集，top-score 必须绑定完整 score table 并重算动作级 entropy；
- state-pool、评测和Position的本地推理服务必须在消费者运行时仍为对应wrapper的存活子进程，
  且实际argv/端口/runtime/模型字节与清单一致；
- M1 reference与实验组按完整selected games而非仅valid rows检查隔离；零范数cosine不得伪报0；
- M2、Position、SAGE的game-cluster区间至少需要两个game；所有可由records/traces导出的计数、
  inclusion probability与success必须重新计算；
- SAGE主报executable-turn条件下的\(U_g\)，并在共同game support上报告A3−A1 Strong差；
- 最终交付包从Git受控文件生成，不包含`.git`、cache、临时输出或密钥，并在独立目录复验。
