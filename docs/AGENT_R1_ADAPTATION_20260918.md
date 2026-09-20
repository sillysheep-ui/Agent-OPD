# Agent-R1 作为 ALFWorld/OPD 候选底座：固定版本适配审计

日期：2026-09-18。此文只记录代码审计与非确认性烟测，不是论文实验结果，也不等于
Agent-R1 已能训练本论文的 OPD。持续问题编号见 [EXPERIMENT_ISSUES_20260917.md](EXPERIMENT_ISSUES_20260917.md)。

## 隔离身份与已有证据

- Agent-R1 `opd`：`7044f15fa19d2505581d0ecd207de8dcf3f2bd6b`；其固定的
  `fishsure/verl` fork：`5779c7c6782733f77ef640f557bea572dfeacc12`。
- 服务器只读源码分别在 `/data/yangchunyu/ld/agent_r1_opd`、
  `/data/yangchunyu/ld/agent_r1_verl_opd`，原官方 veRL 0.8.0 和本仓独立保留。
  借已有 `omniopd-verl080:alfworld` 镜像做无 GPU、无网络、源码只读挂载的 CPU 测试，
  并未将 Agent-R1 与官方 veRL 的包混装。Agent-R1 对官方 0.8.0 的导入曾失败。
- 审计结果目录 `/data/yangchunyu/ld/omniopd_runs/agentr1_audit_20260918/`，包括
  `one_game_scripted_50steps.json`、`student_action_format_vs_agentr1.json`、
  `default_teacher_layout_mask_probe.json`、`loss_alignment_eos_mask.json`。
  仓库含复核脚本，但上述服务器 JSON 尚未归档到 Git；使用证据前需复核文件及 hash。

## 代码变更的最小边界

| 层 | 原始源码/问题 | 拟采用的修改与验收 |
|---|---|---|
| SFT 初始化 | Agent-R1 有 `sft_loss` 函数，但当前固定版本无已验收的本论文加权 SFT recipe。 | 使用本仓独立 veRL SFT 入口先训练共同 Qwen3-4B 初始化；给每臂存同一个 checkpoint 树 hash。若转换格式，逐层核对参数。此项不是让 Agent-R1 原版承包全部训练。 |
| 环境及 Student Flow | `recipes/alfworld/alfworld_agent_flow.py` 默认 Hermes `env_step`，50 步真实环境仅用脚本可运行；真实 SFT Student 生成的是单行 `Action:`。 | 扩展/替换 Agent-R1 Flow 与 prompt 模板，保持同一个 `P_S(s)`、动作解析、50 步、游戏文件和环境种子；对每个回合记录状态/游戏/动作/合法性和实际环境反馈，逐游戏与本仓协议对拍。 |
| 无效动作与终局 | 原 Flow 对不在 admissible 集合的命令仍调用环境 `step`；结束时重复 Student 的末步 prompt/response。 | 冻结 invalid/预算政策后，只执行协议允许的动作；一次 Student 尝试只计一次 token loss，终局奖励必须挂到原步骤或不训练的事件。以唯一 turn/token ID 断言无重复。 |
| 独立 Teacher | `agent_r1/agent_flow/agent_flow.py` 的 `AgentFlowManager._compute_teacher_logprobs` 原样发送 Student `input_ids`；fork 的 `teacher_manager.py` 要求回传 Token 串完全相同。 | 增加可配置的 OmniOPD manager（原 trainer 在 `agent_r1/trainer/ppo/ray_trainer.py` 硬编码实例，须做最小入口改造），从受审计的同一状态构造 `P_T(s)`，追加原 Student response IDs，只按 Student 实际动作 Token 取 Teacher 条件对数概率。核对 Tokenizer ID 语义，拒绝 prompt、动作或路由 identity 不一致。 |
| Teacher 提取与训练掩码 | fork 的 vLLM `_extract_prompt_logprobs` 在目标 Token 缺失时取其他 Token；manager 按 `response_mask.sum()` 补齐；损失的 NestedTensor 切片按同一数量取尾部，EOS 被错误纳入。 | 目标动作 Token 分数缺失或非有限则报错；依据实际 `attention_mask`/response 长度做布局，依据 prompt 和 response 原始边界切取 action scores，`response_mask` 仅决定哪些位置反传。完成有 EOS、左右 padding、多 game、不同长度的等值/梯度对拍，禁止默认分支静默评分。 |
| 研究目标和预算 | Agent-R1 通用 `k3`/RL 组合不等于本论文的 Teacher 动作监督与逐 game 加权更新。 | 传统逐 Token OPD、OmniOPD 有效 Teacher 动作加权 CE 分开实现、分别做梯度单元测试；记录 Teacher 请求、评分 Token、无效尝试、优化步和 GPU 时长；预先冻结比较口径，不以相同请求数冒充同等算力。 |

## 已验证的限制和下一关口

1. 真正的 ALFWorld 游戏文件经 Agent-R1 环境包装器脚本运行 50 步并结束，但没有
   Student 模型参加、没有通关；这只验到环境接口。
2. 真实 Qwen3-4B Student 的一条烟测生成 `Action: go to countertop 2`，在固定版本
   Agent-R1 原 parser 和 fallback 中都得不到 `env_step`，所以不能直接接入。
3. 原 manager 假 Teacher 合同测试中，Student 序列宽 6；将 EOS 置零掩码后 Teacher
   输出宽 7。fork 损失切片的独立 CPU 测试中，目标动作分数 `[11,12]`，读取为
   `[12,99]`，99 是 EOS。**已确认存在公式不符的评分位置错误**。
4. 当前官方 veRL v0.8.0 路径的本仓离线回归为 156 passed、0 skipped；Agent-R1
   这条候选路径仅完成上述检查，**未**运行 Student 真实闭环、Teacher 独立提示的服务调用、
   OPD 参数更新、checkpoint 或多游戏评测。两条路径不得互相借用通过声明。

下一步应先确定无效动作及传统 OPD 预算/目标，再在隔离的 Agent-R1 fork 工作分支中
同时修 Flow、Teacher layout、loss slice 和缺分 fail-closed。先让一局真实 Student
闭环逐步产生可复核轨迹，再跑一条真实 Teacher 分数对拍和可检查梯度的单步训练。
任何完整试验都必须从共同 SFT、同一正式游戏池与新的 provenance 清单重新开始。
