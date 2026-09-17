# Agent OmniOPD 实验问题与跨实验复用记录（始建于 2026-09-17）

本文件记录本地代码迁移、服务器环境搭建和传统逐 Token OPD 技术烟测中**已经观察到**的
问题，以及尚未关闭的实验设计关口。它不是实验结果报告，也不表示问题清单已经穷尽。
原 Word 代码的逐项数学与实现缺陷见 [AUDIT.md](AUDIT.md)；新版 veRL 接口及烟测细节见
[VERL_V080_MIGRATION.md](VERL_V080_MIGRATION.md)。服务器 JSON 产物的路径仅作证据索引，
它们尚未随本仓库一同归档。

本文件是**持续维护**的本地问题台账，不是一次性总结。后续在本项目运行、迁移、复现、
对照或机制分析中发现的问题，都应追加到同一文件；新证据可以更新状态，但不得抹掉
先前的失败现象和当时的代码/环境身份。复用到其他实验时，要先判断模型、数据、框架
版本和研究目标是否相同，不能照搬某次修复命令或把旧烟测当成新实验的验收。

状态含义：**已处理**＝对应技术问题有修复与有限核验；**待验收**＝已有实现但未通过完整
目标路径；**待决策**＝需要先冻结实验口径；**持续风险**＝不能由一次烟测排除。

## 维护规则：后续问题如何同步进来

1. 每发现一个新问题，先分配不复用的 ID（环境/资源 `E`、数据/身份 `D`、方法/代码
   `O`；其他主题可新建前缀），写明发现日期、实际环境/代码提交、**观察到的现象**、
   对实验结论的影响和当前状态。不要把推测的根因写成已确认事实。
2. 同一问题的新排查结果追加日期和证据；区分“绕过报错”“单状态/合成数据核验通过”
   与“完整目标路径通过”。关闭问题时必须留下可复核的测试或产物、适用范围及残余风险。
3. 将可复用的防范方法写成**检查条件**，不要只记录针对这台服务器的一条命令。
   路径、镜像、GPU 和模型版本属于实例证据，下一实验须现场重新确认。
4. 不记录凭据值、个人密钥或不必要的私有数据。大模型、原始数据和大日志只记录受控
   存放位置、内容摘要及访问方式；文档中的服务器路径不等于证据已随 Git 归档。
5. 在本仓库提交文档和更新 `CODE_INVENTORY.json`；若之后执行已授权的服务器或 GitHub
   同步，应把文档提交一并带上，并复核目标分支哈希。**本地记录不自动等于远端已同步**。

新问题的最小记录字段：`ID / 发现日期 / 环境与代码身份 / 现象和复现条件 /
对实验或公式的影响 / 根因（已证实或待证实） / 处理方案 / 验证证据 /
适用范围与残余风险 / 当前状态与下一行动`。若某字段暂缺，明确写“未知/待验证”，
不要为凑齐字段猜测。

## 一、环境、代码同步和运行资源

| ID | 观察到的问题与影响 | 处理与当前状态 |
|---|---|---|
| E01 | 服务器访问 GitHub 的 HTTPS 曾出现 TLS 中断、超时、403 和 shallow clone pack 不完整，导致 veRL 源码获取失败。 | 改用成功的 SSH clone 获取 veRL v0.4.1；之后将 v0.8.0 放在独立 checkout。代码更新在网络不稳时使用经 `git bundle verify` 校验的增量 bundle 和快进合并。**已处理，但网络仍是持续风险**。 |
| E02 | 服务器上的早期 `agent_omniopd` 目录不是 Git 仓库；另一仓库副本又停在旧提交。曾出现 bundle 在共享文件系统上 `index-pack` 失败、临时 worktree 处于无提交 `master` 的情况。单看文件名不能证明实际运行的是哪版代码。 | 以独立 Git 工作树和提交哈希核对；当前服务器副本位于 `/data/yangchunyu/ld/agent_omniopd_v080`，本记录编写前为 `664a80f`。每次同步先检查 clean 状态、验证 bundle，再 `merge --ff-only`。**已处理；GitHub 尚未同步此迁移分支**。 |
| E03 | 宿主 Conda 环境缺 `tensordict`、`hydra`、`torchdata` 等依赖；“宿主能导入”与“Docker 能运行”被混为一谈。现成 `easyr1` 镜像的 NumPy 2 与旧 `pyarrow` 二进制不兼容，且 `pip check` 曾被一个包的异常 wheel tag 阻断。 | 没有继续改动其他任务使用的镜像；基于已有训练镜像构建隔离的 `omniopd-verl041:base`，再为 v0.8.0 建独立 `omniopd-verl080:alfworld`。旧镜像还遇到 NumPy 1.26 与 OpenCV 4.12 依赖冲突，调整后 `pip check` 通过。**已处理于指定镜像；不能推广为服务器全部环境健康**。 |
| E04 | veRL 与本仓同时有 `scripts` 包，`PYTHONPATH=/opt/verl:/opt/agent/src` 时 Python 优先载入 `/opt/verl/scripts`，令本仓测试报 `ModuleNotFoundError: scripts.analyze_m1_gradients`。之后又缺 `pytest`、`codetiming` 等测试/导入依赖。 | 本仓增加显式 `scripts/__init__.py` 并完成依赖镜像构建；新版分支最近一次离线回归为 **156 passed、0 skipped**，Ruff 全过。**已处理**。 |
| E05 | 只读挂载仓库时，Ruff 试图在仓库内创建 `.ruff_cache` 而失败。 | 检查时将缓存放在容器可写位置；这是缓存权限问题，不是代码检查失败。**已处理**。 |
| E06 | 空闲 GPU 0–6 会被环境中的自动填充任务占用；GPU 7 另有约 74 GiB 的既存服务。若仅按 `nvidia-smi` 看到占用就启动或清理，可能干扰别人的作业。 | 在核对进程命令后，按用户此前授权通过 `demokill` 清除匹配的填充任务；后续单卡烟测只用 GPU 0，结束复核显存回落。**持续风险**：自动填充可能再次启动；GPU 7 不属于本实验可清理范围。 |
| E07 | 历史命令文本曾包含 W&B API key。聊天/终端粘贴中的凭据不应进入代码、日志或清单。 | 本记录不复写密钥；若该密钥仍有效，应由持有人轮换并改用受保护的环境注入。**待决策/操作**。 |

## 二、数据来源、版本身份和旧结果

| ID | 观察到的问题与影响 | 处理与当前状态 |
|---|---|---|
| D01 | 旧 `/root/data/alfworld`、`alfworld-prompts` Arrow 数据和旧 `selected_turns.jsonl` 不能自动证明是当前 TextWorld 可执行游戏或同一协议数据。 | 已从 ALFWorld 官方资源重建并校验压缩包、环境和一局 reset/step；详见 [ALFWORLD_REBUILD_20260916.md](ALFWORLD_REBUILD_20260916.md)。该检查**不是**完整 Student 轨迹验收。正式 Student state pool 与选择清单仍须重建。**待验收**。 |
| D02 | 旧版 veRL 0.4.1 行动模仿链、当前 v0.8.0 迁移链与传统逐 Token OPD 是三种不同路径。原仓自有 FSDP trainer 做加权 action CE，不能因为位于 veRL 目录就称为 OPD。 | v0.8.0 独立 checkout 的提交和包版本已核对；4 卡合成数据单步 LoRA 保存—加载通过，但只证明 SFT 路径技术可用。**逐 Token OPD 仍待验收**。 |
| D03 | 旧审计中的 state/history、Teacher prompt、训练 mask、权重、评测身份等缺陷使历史结果不能只靠新训练器或新报告修补。 | 原因和修复见 [AUDIT.md](AUDIT.md)；旧确认性数据不复用。代码 revision 改动后，正式 pool、selection、训练及评测产物必须从头生成。**待重跑**。 |
| D04 | 当前 OPD 单状态烟测使用旧版 pool 中一条真实 ALFWorld 状态，虽然状态真实，但其来源 revision 不满足新版正式清单的代码身份合同。 | 所有相关输出均标为 nonconfirmatory；`build_opd_prompts.py` 的清单明确 `training_ready: false`。**待重建正式输入**。 |

## 三、传统逐 Token OPD 的代码与公式契合性

| ID | 观察到的问题与影响 | 处理与当前状态 |
|---|---|---|
| O01 | Qwen3-4B SFT Student 与 Qwen3-14B Teacher 的 chat template 不同：Student assistant 头后直接生成动作；Teacher 的 `enable_thinking=False` 模板先放置闭合的空 thinking 段。若强求两侧 prompt Token 序列相同，会错误否定可比较性；若把 Student 的 `P_S` 原样给 Teacher，又违反论文的 `P_T` 定义。 | 已审计相同 Token ID 语义、特殊 Token 和编码规则，分别使用 `P_S(s)` 与 `P_T(s)`，再让 Teacher 评分**同一组 Student 动作 Token**。这是提示词语义对齐，不意味着两模型分布或输出相同。**已处理于单状态接口**。 |
| O02 | veRL 默认 Teacher 评分使用 Student `prompt_ids + response_ids`，无法自动实现本研究独立的 Teacher 系统提示词。 | 自定义 `OmniOPDAgentLoopWorker` 读取经过验证的 `teacher_prompt`，核对两侧共享同一状态历史，再请求 Teacher。真实状态/Tokenizer 的 Worker 合同测试通过；完整 Ray/Teacher 服务尚未跑通。**待验收**。 |
| O03 | veRL `extract_prompt_logprobs` 将下一个 Token 的分数放在当前位置，训练端 `no_padding_2_padding` 又从 Student 提示词最后一位开始取响应分数。最初直接按未左移位置映射会造成首、末动作 Token 错位。 | 已修正重映射；模拟 Teacher 返回＋真实 veRL 切片、真实 vLLM Teacher 返回＋真实 veRL 提取器均通过位置检查。EOS 分数置零。**已处理于单状态接口**。 |
| O04 | Student 实际 vLLM 输出是否包含 EOS 曾未验证；若不包含，当前严格动作解析会拒绝响应，影响 rollout 预算与训练。 | 在同一真实状态，vLLM 0.8.5 生成 `Action: go to countertop 2<|im_end|>`，`finish_reason=stop`；将原始 Token 输入自定义 AgentLoop 后动作 mask 为 1、EOS 为 0。**单状态已处理，其他状态/截断情形仍是持续风险**。 |
| O05 | Qwen3-14B 在同一 8 个动作 Token 上，HF 与 vLLM 的 Teacher logprob 前 7 位接近，末位分别为 `-17.0` 与 `-16.75`，最大绝对差 `0.25`。 | 真实 vLLM 返回经过 veRL 原生提取器后与直接读取 vLLM 的分数逐项一致，因此位置/提取结构已核验；**数值差异的原因未确定**。不得声称两后端完全一致或擅自归因为 BF16/特定算子。 |
| O06 | veRL sampled-token `k1` 若直接反传，其梯度不携带 Teacher 信息；仅有 Teacher logprob 张量不保证实现了合理的 OPD 更新。 | 当前 CPU 合同测试调用 veRL 原生 `k3` 估计器，验证有效动作位梯度方向与 EOS 零梯度；这使用人工 Student 分数，**不是**真实模型反向传播。正式训练须冻结损失模式、梯度路径和优化超参。**待验收**。 |
| O07 | 当前严格 parser 对不在 admissible 集内、非单行动作、含特殊/思考 Token 或未以 EOS 结束的 Student 输出直接报错。若整批失败并免费重采样，会改变 Student on-policy 样本分布及预算。 | 一条有效动作只证明 happy path；必须预先规定 invalid、长度截断与异常的记录、预算和训练处理，不得静默丢弃或免费补打。**待决策，确认性训练阻断项**。 |
| O08 | 固定 Student state pool 上的逐 Token 更新，是**条件于已选状态**的 on-policy action-token 蒸馏，不是训练中实时交互形成的新状态分布；它与 OmniOPD 的 Teacher 重采动作/有效性筛选并非相同监督单位。 | 报告必须分别说明 state-source、Token-prefix 与动作来源；不得把固定池 comparator 写成全在线 agent rollout。**待冻结论文表述和实验协议**。 |
| O09 | OmniOPD 的 150 次 Teacher action attempts 与传统 OPD 的逐 Token Teacher 评分不能只按“请求次数”直接视为等预算；长度、模型大小和推理后端都会改变成本。 | 正式对照前应预定义至少 Teacher 评分 Token 数、Student 状态/动作数量、优化步数和时间/算力口径，并将 invalid attempt 计入相应预算。**待决策，确认性比较阻断项**。 |
| O10 | 现有检查尚未调用完整 veRL Ray Teacher 服务链、未对真实 Student 参数执行 OPD 单步反向/保存/重载，也未生成正式 launch/completion manifest。 | 单状态 HF/vLLM 前向、AgentLoop、Worker、损失张量测试**不能**替代训练验收。下一技术关口是完整服务调用与单步 checkpoint 验证；在此之前 `training_ready=false`。**待验收，确认性训练阻断项**。 |
| O11 | OPD prompt builder 将逐 game 的 `state_weight` 写进 `extra_info`，但当前自定义 AgentLoop/Worker 与已测试的 veRL `k3` 损失没有消费此字段。仅有元数据不能证明训练目标按 game 加权。 | 若正式设计要求 game-balanced 目标，必须在实际损失或经证明等价的采样器中接入权重，并做多 game、不等状态数的梯度测试；若每 game 固定相同状态数且目标是均匀状态均值，也应明确证明该等价条件。**待决策并待实现/验收**。 |
| O12 | `build_opd_prompts.py` 目前输出经审计的 JSONL 提示词行和 `training_ready=false` 清单；还没有经过正式 veRL 数据加载、训练配置、启动与完成清单的全链验证。不能把“已构造 prompt 行”解释为“有可运行的 OPD 数据集”。 | 在冻结无效处理和权重口径后，增加受审计的数据加载/转换及一条完整的 veRL 技术启动链，并验证行数、state 哈希、`extra_info` 透传与产物身份。**待实现/验收**。 |
| O13 | “同一模型在不同 ALFWorld 框架的准确率不同”不能直接归因为模型变化。AgentBoard 的任务代码默认 `max_num_steps=30`，本仓协议为 50；它还使用自己的示例提示词、动作解析与成功/进度/grounding 记录路径。游戏集合、`done`/`won` 语义、Token 模板及解码参数是否一致需要逐项核对。 | 跨框架报告先区分成功率、进度率和动作 grounding，不按指标名称猜测同义；冻结同一游戏文件及 split、逐 game 种子、最大步数、prompt/历史截断、动作解析/无效动作、模型与 Tokenizer 身份、推理后端和采样设置，再做逐游戏配对对拍。**已确认存在协议设置差异；它们对具体分数差的贡献仍待实测**。来源：AgentBoard `agentboard/tasks/alfworld.py`、`assets/agent_customization.md`，本仓 `configs/alfworld_textworld.yaml`、`src/omniopd/adapters.py` 与 `src/omniopd/protocol.py`。 |
| O14 | Agent-R1 现有 ALFWorld 的多步环境流转和另一个 `opd` 分支的通用蒸馏入口，可能减少本仓自建在线轨迹调度代码；但 `main` 的 ALFWorld recipe 与 `opd` 分支的 GSM8K OPD 示例并不是一个已验证的“ALFWorld+OPD”组合。`opd` 分支固定依赖 `fishsure/verl@5779c7c6782733f77ef640f557bea572dfeacc12`，不能直接假设与当前官方 veRL v0.8.0 接线兼容。 | **待架构评估**：在独立 checkout/镜像中固定 Agent-R1 的提交，核对 ALFWorld `reset/step`、状态提示词、动作解析、轨迹掩码和奖励；核对 OPD Teacher 是在本研究独立 `P_T(s)` 下对 Student 实际动作 Token 评分，及无效尝试、状态加权、预算与 checkpoint。当前 veRL v0.8.0 单状态接线保留作对拍基线，不把两个分支或版本混装。来源：Agent-R1 官方 `README.md`、`recipes/alfworld/README.md`、`examples/gsm8k/run_opd.sh`、`requirements-opd.txt`（2026-09-17 查阅）。 |
| O15 | Agent-R1 的公开训练说明主要是多步 RL 和 `opd` 分支的通用 Token 蒸馏；未找到可直接替代本研究冷启动 SFT/有效 Teacher 动作加权 CE 的 Agent-R1 官方 SFT recipe。把“Agent-R1 可收集轨迹”推断成“已原生支持本论文两类 SFT”会误设实施范围。 | **待接口验收**：优先复用已单步验证的本仓 veRL SFT 路径训练共同冷启动 Student，再将其 checkpoint 接入 Agent-R1；所提方法的 Teacher 动作监督、有效性过滤和逐 game 权重须作为独立的加权 SFT 数据/损失链实现。若改为 Agent-R1 内部 SFT，先证明与现有训练器的 token mask、权重、优化及 checkpoint 可比。veRL 官方提供 SFT trainer；Agent-R1 原生 SFT 支持范围仍需在固定源码提交下复核，不能声称已证明不存在。**待决策/验收**。 |

## 证据索引与下一步顺序

- 本地代码：`src/omniopd/opd_adapter.py`、`src/omniopd/verl_opd.py`、
  `scripts/check_verl_opd_worker.py`、`scripts/smoke_opd_vllm_student.py`、
  `scripts/smoke_opd_vllm_teacher.py`。
- 服务器非确认性输出根目录：
  `/data/yangchunyu/ld/omniopd_runs/opd_preflight_20260917/`；其中
  `smoke_teacher_scores_state0_seed42.json`、
  `smoke_vllm_student_state0_seed42.json`、
  `smoke_vllm_vs_hf_verl_extractor_state0_seed42.json` 分别记录 HF 前向、
  vLLM Student 终止和真实 vLLM Teacher/veRL 提取器核验。不同烟测生成于
  不同代码提交，**不可**拼成同一份正式 provenance 清单。
- 下一步按顺序：先冻结 O07/O09 的实验口径；再做 O10 的完整服务＋真实单步
  梯度/保存/加载烟测；随后按同一冻结 revision 重建 D01/D03/D04 的正式
  state pool 和所有下游输入；最后才启动匹配的确认性训练与评测。

本记录更新不等于服务器同步或 GitHub 推送；任何执行者仍须现场核对 Git 提交、
镜像 ID、模型/Tokenizer 身份、GPU 占用、数据哈希与实际输出。

## 跨实验启动前的复用检查

| 检查条件 | 本次暴露的误判 | 新实验所需的独立证据 |
|---|---|---|
| 隔离环境并核对实际解释器/容器 | 宿主有包不代表容器有；`pip check` 通过也不代表二进制导入可用。 | 镜像 ID、容器内版本/导入、依赖检查和最小真实模型加载。 |
| 锁定源代码与数据实体 | 相同目录名、模型 alias 或 manifest 自述不等于同一份内容。 | clean Git 提交、文件内容哈希、模型/Tokenizer 指纹及消费者重新验算。 |
| 区分技术烟测和论文结果 | 一局环境 reset、一次有效动作、一次合成单步训练都只覆盖局部路径。 | 与研究主张对应的完整数据、训练、评测和多 seed/game 审计。 |
| 核对公式在实际梯度中的路径 | 有 `state_weight` 字段、Teacher 分数张量或 KL 名称不等于目标权重/梯度已生效。 | 真正损失调用、多样本/不等权重梯度测试及单步参数更新。 |
| 预先定义无效输出与成本 | 丢弃失败样本或免费重采样会改变 on-policy 分布；请求数不能代替 Token/算力预算。 | 不可篡改的 attempt/Token ledger、明确的 invalid 政策和比较口径。 |
| 不碰未授权资源 | GPU 占用可能是填充任务，也可能是他人服务。 | 进程归属核对、明确的设备范围、运行前后资源记录。 |

更新记录：2026-09-17 建立 E01–E07、D01–D04、O01–O12；同日补充持续维护规则和
跨实验复用检查。2026-09-17 追加 O13（跨框架 ALFWorld 评测可比性；来源为 AgentBoard
官方代码及本仓协议，具体分数归因尚未验证）、O14（Agent-R1 候选路线及版本兼容风险）
与 O15（Agent-R1/veRL 的 SFT 职责边界）。
同日完成当前 veRL v0.8.0 单状态非确认性输入生成、156 项离线测试和 Hydra 配置解析；
**尚未**启动 Ray/vLLM 完整单步反向训练。后续更新应在此处追加日期、涉及 ID 和证据，
不覆盖旧条目。
