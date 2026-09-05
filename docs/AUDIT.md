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
- uncertainty score 强制使用生成该 Student-state pool 的同一模型与Tokenizer内容指纹，
  防止把其他checkpoint的不确定性误标为当前Student uncertainty。
- 训练以 uniform-row 抽样的期望质量 `rW/D` 归一，microbatch 只累加 numerator；FSDP
  padding 权重为 0，optimizer step、precision、veRL 版本和四个 training seeds 显式冻结。
- 评测结果必须绑定 final `global_step`、training manifest、base/LoRA 内容指纹、冻结 game
  list manifest、稳定 served-model alias、vLLM 版本和独立 rollout seed；推断同时重采
  配对的 training seed 与 game。
- M3 先冻结跨 game、同 task、action-admissible、未被任何 design 选中的邻居 panel，再用
  共同 base 与两个由规范训练 manifest 绑定的最终 adapter 计分；M1输入也绑定完整未划分
  action-table audit；SAGE blind judge 不接收 Teacher/entropy/group，
  失败标签保持 missing；Position 从完整历史恢复 (P_S) 后对称 replay 两个动作分支。

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
单变量设计约束模板，只有完整填入模型/hash并完成新实验后才能形成 state-source 结论。

## 旧代码处置

`legacy/main_document` 与 `legacy/supplement_document` 是不可变审计快照。它们用于
定位历史产物，不是 canonical runtime。新实验只允许导入 `src/omniopd`，运行时应记录
模块真实路径与 SHA256，防止 `/cfs`、`/root/code`、`/root/data` 多代码根漂移。
