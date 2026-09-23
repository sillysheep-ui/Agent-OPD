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

6. 每项任务只在一个**专用分支**上推进，不把任务提交直接堆到共享集成分支。分支名需表明
   任务范围（例如 `feat/expert-sft-coldstart`）；同一任务需要拆分主题时，先在同一分支上
   分提交，确实需要独立分支时再从该分支派生，不要平行创建多个同源分支。
7. 服务器同步只通过 `git bundle` 传递已提交的引用，不靠复制工作区文件。同步前先确认
   服务器工作区干净；若服务器上存在未提交文件，先核对它的 blob 是否已进入 Git，再把它
   移到备份目录（例如 `/data/yangchunyu/ld/untracked_backup_20260920`）而不是直接删除。
8. 每次同步后必须在容器里重跑离线回归，并把通过数写进本文件。回归数字、镜像标签和
   分支提交三者缺一，不能声称该版本已验证。

### 当前任务分支（2026-09-20）

| 分支 | 任务范围 | 状态 |
|---|---|---|
| `feat/expert-sft-coldstart` | 共同冷启动 SFT：ALFWorld handcoded expert 采集、通用 SFT 目标 schema、Qwen3-14B 教师 pilot 脚本留存 | 本任务的单一专用分支；容器回归 158 passed |
| `feat/verl-v080-opd` | veRL v0.8.0 迁移与逐 Token OPD 对照 | 既有共享集成分支；含 3 个本会话早期提交 |
| `main` | GitHub 主干 | 与 `origin/main` 一致；未合并上述任务分支 |

`feat/expert-sft-coldstart` 起于 `feat/verl-v080-opd` 的 `13039e9`，因此包含该集成线截至
该提交的全部历史。任务完成后由人工决定如何并入 `main`，不由自动化流程代劳。

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
| O16 | 固定的 Agent-R1 `opd@7044f15` 对应 `fishsure/verl@5779c7c`，其 `AgentFlowManager` 默认发送完整 Student `input_ids` 给 Teacher，fork 又硬性要求返回的 Token ID 与请求逐位相同；直接替换为独立 `P_T(s)` 不符合代码合同。 | 在隔离路径 `/data/yangchunyu/ld/agent_r1_opd` 与 `agent_r1_verl_opd` 完成源码核查和容器 CPU 导入；原版 Agent-R1 配官方 veRL 0.8.0 的导入已失败。**待修改/验收**：在固定 fork 中接入可配置 manager/Teacher prompt，给同一 Student 动作在 `P_T(s)` 下评分，并明确两侧 Token ID 语义及逐位置对齐；不得把单纯换提示词当成完成。 |
| O17 | Agent-R1 ALFWorld 默认输出 Hermes `env_step` 工具调用，而现有 SFT Student 的真实烟测输出为 `Action: go to countertop 2`；默认 parser 和文本后备都提取不到动作。其 recipe 默认每局 20 步，而本论文协议为 50 步。 | 真实 Student 输出与固定 fork parser 的 CPU 比对为 `directly_compatible=false`；只读一局真实游戏的 50 步环境包装器运行结束，但未获胜，也没有使用模型。**待修改/验收**：自定义 Agent Flow 保留原有 `P_S`、`Action:` 输出、50 步上限、游戏/环境种子、状态历史和无效动作政策；再做真实 Student 闭环逐游戏对拍。 |
| O18 | Agent-R1 ALFWorld Flow 在命令不在 admissible 集合时仍调用 `executor.step(command)`；`done` 后又追加一次相同 prompt/response 的 `final_step` 以挂终局奖励。若 OPD 训练消费所有步骤，最后一段动作的 token 更新可能重复；后者仍需在训练批次确认。 | 源码直接证实执行/追加行为；待通过 Worker 的批次产物确认重复更新范围。**待修改/验收**：无效命令先按冻结政策计费/记录并禁止意外环境推进；终局奖励应挂到原动作步骤或用明确零训练 mask 的奖励事件，按 state/turn/token ID 查重。 |
| O19 | Agent-R1 Teacher 布局按 `response_mask.sum()` 推断右侧 padding；保留在 `input_ids` 内但不给 EOS 训练权重，会生成比 Student 序列更宽的 Teacher 张量。fork 的 NestedTensor 路径还按 mask 有效数量从整段序列尾部取分数，**已证实**将 EOS 评分替代首个动作评分。 | CPU 假 Teacher 合同测试：6 Token Student、完整 response mask 时 Teacher 宽 6，动作-only mask `[1,1,0]` 时宽 7；固定 fork 的真实损失切片诊断中目标动作分数 `[11,12]`，却读为 `[12,99]`（99 为 EOS），证据 `loss_alignment_eos_mask.json`。**待修改/验收**：布局/切片必须使用实际 attention 与原始 response 长度定位评分，用训练 mask 仅做加权；新增含 EOS/左右 padding/多样本的梯度与张量对拍。 |
| O20 | `agent_r1/workers/utils/losses.py` 定义了 `sft_loss`，但 Agent-R1 现有 recipe/engine 没有找到可直接运行本论文冷启动或加权有效动作 SFT 的入口；“有函数”不等于“有训练管线”。 | 固定源码的调用搜索只有定义/导出，engine 选择 PPO/蒸馏损失；继续沿用本仓已验证的独立 veRL SFT trainer，先核对相同模型、action mask、样本权重和产物身份，再进行共同初始化。**待验收**，不得宣称 SFT 已完成。 |
| O21 | 固定 fork 的 `_extract_prompt_logprobs` 在目标 Token ID 缺失于 vLLM 返回字典时，取字典首项的其他 Token 分数；若该位置是 `None` 还写入 `0.0`。Teacher token ID 列表即使逐位相同，也不能单独证明对应 logprob 是该 Token 的真实评分。 | 源码核对：`verl/workers/rollout/vllm_rollout/vllm_async_server.py` 的提取器，与 `teacher_manager.py` 仅比较 `teacher_ids` 的断言。**待修改/验收**：对所有纳入损失的 Student 动作 Token，字典必须包含目标 ID 且分数有限；否则 fail closed 并记入错误/成本 ledger。仅允许第一个无条件 Token 等预先定义且完全不入损失的位置缺分。 |

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
- Agent-R1 分支专项（2026-09-18）：先修 O17/O18 的 Student 动作与环境事件协议，
  再修 O16/O19 的独立 Teacher 打分、EOS 与逐 Token 对齐；用合成序列证明梯度只落在
  动作 Token 后，再启动真实 Student 轨迹＋单步训练/保存/加载。它与上文当前 veRL
  v0.8.0 路径是**隔离的候选实现**，不是已经通过的替代品。
- Agent-R1 原版固定 checkout：`AgentR1/Agent-R1` 的 `opd@7044f15fa19d2505581d0ecd207de8dcf3f2bd6b`
  与 `fishsure/verl@5779c7c6782733f77ef640f557bea572dfeacc12`；2026-09-18 的
  只读检查产物位于 `/data/yangchunyu/ld/omniopd_runs/agentr1_audit_20260918/`。
  路径是证据索引，不能替代归档文件与 checksum。

## 2026-09-18：Qwen3-4B/14B 配对基础测试新增问题

- **E08｜推理服务中途消失；根因待证实。** 首次尝试将 14B 与 4B 暂时放在 GPU 0，
  14B 首次请求得到 `openai.APIConnectionError`，随后只读检查发现两个临时 vLLM
  容器以及先前的 GPU 服务均不在运行，所有 GPU 显存回落。容器当时设置了 `--rm`，
  未保留服务退出日志；不能据此断言是显存不足、自动填充任务、外部清理或模型问题。
  用户随后明确指定改用 GPU 6/7。重试时将 14B/4B 分别放在 GPU 6/7，取消
  `--rm` 以保留日志，两服务就绪后才发起请求，完整试跑退出码 0。两测试容器
  在结束后被停止，GPU 6/7 显存均回到约 1 MiB。**当前状态：服务迁移绕过故障；
  首次退出根因未证实。** 后续实验应保留容器日志并记录 GPU/容器事件，不应把
  API 连接失败自动重试成免费的教师调用。
- **E09｜共享文件系统上的 ALFWorld 全量游戏枚举与元数据读取很慢。** 首次
  配对脚本在实际推理前耗时约十余分钟；`AlfredTWEnv` 扫描 8810 个路径需
  4 分 26 秒，随后逐个读取训练游戏的 `traj_data.json` 继续等待共享文件系统。
  首次输出已固定六个新训练游戏的 `game_list.json`；第二次使用同一清单运行，
  跳过全量扫描，约数分钟完成完整配对。**当前状态：本次绕过；持续风险。**
  后续先生成带数据身份哈希的不可变游戏清单，并由训练/评测共同消费，避免
  每次启动重复扫描，也避免游戏顺序漂移。
- **D05｜非确认性配对数据身份。** 结果位于
  `/data/yangchunyu/ld/omniopd_runs/paired_models_20260918_02/`；两臂均使用
  `game_order_seed=42` 的每类第 2 个游戏、相同六个 `game.tw-pddl` 文件及相同
  环境种子。独立只读脚本 `scripts/analyze_paired_model_baseline.py` 验得
  `same_game_file_and_environment_seed=True`，14B 的 232 次和 4B 的 300 次
  请求状态均为 `ok`。两臂的 prompt SHA256、脚本 SHA256、4096 上下文、
  50 步上限、温度 0、非思考和 admissible-choice 约束一致；模型权重与模板
  不同，不能把结果解释为单纯参数量消融。**当前状态：小样本技术对照已完成；
  非正式数据。** 六个训练游戏不能估计泛化率，更不能复用为 SFT 数据或确认性
  评测。
- **O22｜合法动作不等于任务推进，也不足以证明教师可靠。** 此六游戏中
  Qwen3-14B 成功 2/6（清洗、双物体放置），Qwen3-4B-Instruct-2507 成功
  0/6；所有动作均合法是 vLLM `structured_outputs.choice` 强制约束的结果，
  不是模型自然遵守格式的证据。失败轨迹出现重复 `look`、往返移动及反复开关
  容器；14B 的加热任务有 42/50 轮丢弃早期交互历史。14B 对 4B 在这六局上
  有观察到的优势，但 2/6 不足以确认它能稳定提供冷启动示范或 OPD 教师反馈。
  **当前状态：待扩样验证。** 在冻结完整提示/历史/动作协议后，应使用新的训练
  游戏分层扩样，报告按任务类型的配对成功、循环、历史截断和调用成本；如比较
  SAGE-OPD 设置，需另行运行其 ReAct/示范/步数协议，不得将本次强制候选动作
  的 50 步结果直接称作该论文复现。

## 2026-09-20：veRL v0.8.0 传统逐 Token OPD 真实单步验收

- **E10｜镜像声明兼容不等于运行时兼容。** 早期现成镜像中的 vLLM 0.8.5
  缺少该 veRL 路径需要的 `run_headless` 接口；另一上游镜像虽带 vLLM 0.10.0，
  但 TensorDict 为 0.9.1，而 `DataProto.to_tensordict()` 运行时断言要求至少
  0.10。当前隔离镜像 `omniopd-verl080:vllm010-td010` 基于固定上游镜像，仅以
  `--no-deps` 将 TensorDict 固定为 0.10.0，并启用 vLLM V1；不得把该修复回写
  其他共享镜像。Teacher prompt-logprob 服务还要求温度为 1.0，0.45 的显存比例
  不足，本次单状态 1152-token 烟测用 0.60 通过。**当前状态：指定镜像和烟测
  规模已处理；扩大上下文、batch、模型或并发后必须重新测量显存。**
- **O23｜veRL v0.8.0 的 K3 hard clamp 可产生“非零 loss、零梯度”的假通过。**
  真实 Qwen3-4B/14B 单状态运行中，K3 内部将大的逐 Token 估计硬截到 10，日志
  loss 非零但 `grad_norm=0`，优化器一、二阶动量全零；仅移除外层 clamp 不能修复
  内部饱和。技术烟测临时切到源码注释所述具有正确期望梯度的 K2，并取消外层
  loss clamp。**当前状态：根因已由临时梯度 hook 证实，工程烟测已绕过；正式
  实验仍须预注册 K2、修改后的 K3/straight-through 或其他估计器，不能把本次
  临时选择直接升级为论文协议。**
- **O24｜Student rollout 温度 0 会破坏该实现中的训练 log-prob 梯度。** veRL
  训练前向复用 rollout 温度并用其缩放 logits；温度 0 被夹到极小正数后令
  softmax 饱和。临时 hook 观察到损失梯度已经到达 Student log-prob，但无法继续
  形成参数梯度。将 Student 温度改为 1.0 后，同一真实状态的 step 1 得到
  `grad_norm=2.5853828992694616e-4`。**当前状态：技术烟测已处理；正式协议必须
  区分训练采样温度与贪心评测温度，并记录实际采样分布。**
- **O25｜传统逐 Token OPD 的真实单步保存—恢复技术关口已通过，但不等于正式
  实验可启动。** 非确认性 step 1 使用 Qwen3-4B Student、Qwen3-14B Teacher、
  K2、Action-only mask 和学习率 `1e-6`，成功保存 model/optimizer/extra state；
  优化器共 1512 个张量，其中 1008 个张量、35,389,942 个元素非零，所有浮点
  张量有限。随后从该 `global_step_1` 显式恢复，日志确认 global step 设为 1，
  step 2 的 `grad_norm=2.3061489628162235e-4`，并成功保存 `global_step_2`。
  证据目录分别为 `/data/yangchunyu/ld/omniopd_runs/opd_real_step_smoke_20260920_11/`
  与 `opd_real_step_resume_20260920_12/`。**当前状态：O10 所要求的真实服务、
  反向、优化器更新、保存和恢复技术验收已关闭；仍为单状态非确认性烟测。**
  O07/O09 的 invalid 与预算口径、O11 的 `state_weight` 消费、正式 estimator、
  多状态/多游戏数据和完整 provenance 尚未冻结或验收。首次 step 的 actor 峰值
  allocated memory 约 30.98 GiB，恢复 step 约 46.95 GiB；增长原因未知，不能仅凭
  两条日志归因于 checkpoint 恢复。扩大 batch、上下文或并发前须重新测量并保留
  每卡峰值，不能据首次运行余量直接估算正式容量。

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
2026-09-18 追加 O16–O21（Agent-R1 固定版本的独立 Teacher prompt、动作协议、
终局重复步、EOS mask、SFT 入口边界和评分 fail-open）；实际游戏脚本只证明环境可运行，尚未证明
真实 Student 闭环或参数训练。此日期的 CPU 诊断不能升级为论文实验结果。
同日完成当前 veRL v0.8.0 单状态非确认性输入生成、156 项离线测试和 Hydra 配置解析；
**尚未**启动 Ray/vLLM 完整单步反向训练。后续更新应在此处追加日期、涉及 ID 和证据，
不覆盖旧条目。
2026-09-18 再追加 E08、E09、D05、O22：记录双模型试跑首次服务消失、共享文件系统
初始化延迟、GPU 6/7 重试及六类新训练游戏的配对结果。仅为推理/交互技术测试，
不代表 SFT、传统 OPD 或 OmniOPD 训练已经启动。
2026-09-20 追加 E10、O23–O25：记录 vLLM/TensorDict 运行时兼容、K3 hard-clamp
零梯度、Student 温度 0 导致 softmax 饱和，以及真实 step 1 更新、优化器非零和
step 2 checkpoint 恢复证据。该更新只关闭 O10 的单状态工程验收，不解除正式
数据、估计器、invalid、预算、权重和 provenance 阻断项。

## 2026-09-20（续）：共同冷启动 SFT 的复用边界与 expert 冒烟

- **O26｜冷启动 SFT 不需要第二套采集、数据构造和启动链，但此前的缺口清单把它估大了。**
  现状是：`protocol.rollout_episode()` 接收任意 `ChatPolicy`，因此新增的
  `ExpertPolicy`（`src/omniopd/adapters.py`）只需把环境 expert 适配成 `generate()`
  契约，history 状态机、task-preserving 截断、state hash 与 provenance 全部留在同一条
  代码路径上；监督行构造复用 `dataset.build_training_rows()` 与 `omniopd-build-data`
  CLI，数据集复用 `torch_dataset.FinalTurnActionDataset`，训练器复用
  `integrations/verl/fsdp_sft_trainer.py`。**待改造**：`run_verl_train.sh` 目前强制
  要求 breadth/depth 的 annotation-pair，冷启动需要加互斥模式；pool manifest 的
  validation 假设 behavior policy 是 API 服务，expert 需要一个独立的身份分支。
- **D06｜论文原始 SFT 实现已在仓库内，但只能作为规则来源与对照。**
  `legacy/supplement_document/build_sft_v6.py`（轨迹→SFT 转换，含 7 条硬规则）、
  `legacy/main_document/sft_teacher_collect.py`、`collect_teacher_rollout.py`、
  `multiturn_sft_dataset.py`、`final_turn_dataset.py`、`run_verl_sft_v6.sh` 都是原始
  实现；其已知缺陷（history 错位、`[:, :-1]` mask 偏移、backward 后再除 microbatch、
  valid 过滤改变 Teacher 分布等）已逐条记录在 [FILE_AUDIT.md](FILE_AUDIT.md)。因此新
  冷启动数据链**不重写**这些功能，而是复用上条所列的现代实现；`legacy/` 仅用于核对
  规则与解释历史结果。**待验收**：handcoded expert 语料仍需按新协议重建并出具 manifest。
- **D07｜expert 采集路径已在真实 ALFWorld 上跑通（非确认性）。** 单局
  `pick_heat_then_place_in_recep`（train split）经 `rollout_episode` + `ExpertPolicy`
  完成 7 步并获胜，`invalid_turns=0`，全程无模型推理、无 API 调用、无训练（`model_used=false`、
  `api_calls=0`、`training_performed=false`）。产物
  `/data/yangchunyu/ld/omniopd_runs/expert_smoke_20260920_1/expert_rollout.json`
  逐轮记录了 state hash、admissible 数量、截断信息、执行动作与有效性，并绑定了
  game 文件与 trial 目录指纹、`derive_environment_seed` 派生的 per-game 环境种子及其
  TextWorld seed attestation、tokenizer 配置哈希。同一提交的容器离线回归为
  **161 passed、0 skipped**，Ruff 通过。**当前状态**：路径可用；多游戏、按 game 划分、
  manifest 与偏好核验尚未执行，仍为非确认性。

## 2026-09-21：参考协议复刻与提示词比选

- **E11｜ALFWorld 数据目录在服务器上被移动。** 原先使用的
  `/cfs/data/private/yangchunyu/ld/alfworld_data_0.4.2` 整个 `yangchunyu/ld`
  前缀在 FUSE 挂载上消失，同一挂载的 `zhangsl/Model` 正常，`/data/yangchunyu/ld`
  下的仓库与产物也正常。实际新位置是
  `/cfs/data/private/yangchunyu/liud/ld/alfworld_data_0.4.2`（多了一层 `liud`）。
  **处理**：所有容器挂载改指新路径；`json_2.1.1/{train,valid_seen,valid_unseen}`
  与 `logic/` 齐全，`valid_seen` 含 140 局、`train` 含 2435 个任务家族目录。
  **教训**：不能把机器特定路径写进代码或脚本，挂载点必须现场确认。
- **E12｜该 FUSE 挂载上的深递归遍历会卡死。** `AlfredTWEnv.collect_game_files`
  先做 `list(os.walk(root))`，在容器内两次阻塞在 `fuse_readdir`（进程 D 状态，
  6 分钟无 I/O 进展）；宿主机对同一棵树 `du -sh` 也超时。单层 `ls` 却很快
  （407 个家族 0.83 s、单家族 0.026 s）。**处理**：新增单层枚举
  `enumerate_split_games` 与游戏清单缓存（`--game-list-cache`），采集/评测共用；
  失败的单局环境加载记为该次尝试的失败原因而不中断整批。
- **E13｜本机 `~/.ssh/config` 损坏。** 报
  `no argument after keyword "host"`，导致所有 ssh/scp 失败。
  **处理**：本次全部命令改用 `ssh -F /dev/null` 绕过；该文件需要使用者自行修复。
- **R01｜SAGE-OPD（arXiv 2606.19659）的实测配置（原文证据）。**
  teacher **未经训练**：Table 1 直接列出 teacher 的 base 推理成绩
  （Qwen3-8B 61.43/67.16、Qwen3-32B 57.14/69.40，seen/unseen SR）。
  off-policy SFT 只作为基线且几乎无效（0.6B 1.43→2.86、1.7B 25.71→25.00），
  而同一对 OPD 把 0.6B 提到 55.00/56.72、1.7B 提到 60.00/61.94。
  主方法训练超参：verl 全异步、LR 1e-6、weight decay 0.1、response 4096、
  rollout n=1、global batch 64、1 epoch；ALFWorld 最多 30 轮、每轮最多 4096 token；
  temperature 0.4、top-p 1.0；评测用 ReAct 格式 + full chat history（无滑窗）+
  thinking 关闭 + `</action>` 停止符，评 valid-seen/valid-unseen。
  其 one-shot 示范**内容未公开**（表头只写 "An one-shot demonstration"）。
- **R02｜按 Table 6 复原的协议未能复现论文的 base 数字。** 用 Qwen3-1.7B 在
  完整 valid_seen（140 局）上，按 Table 6 的 system prompt、`Task/Observation/
  Admissible:` 用户轮（分号分隔、前 30 条 + `(+N more)`）、模型自身 `Thought/Action`
  历史、30 轮、贪心解码，得到 **11/140 = 7.86%**；论文同模型为 **25.71%**。
  追加实验：自建 one-shot 示范（取自 train 的 cool 局，13 轮；因含连续三次
  `examine fridge 1` 而已改选 4 轮无重复的 `pick_and_place_simple` 局）
  + temperature 0.4，在"基线失败的 12 局"上仍是 **0/12**；同一批换 AgentBoard
  形式为 1/12。**结论**：论文未公开的示范内容、observation/历史拼装与解析策略
  使精确复刻不可达；已排除模型变体（所用 `Qwen3-1.7B` 为 instruct、带 chat
  template）与采样温度两个原因。
- **R03｜提示词比选（Qwen3-4B-Instruct-2507，固定 18 局，每类 3 局，30 步）。**
  AgentBoard 形式（引导语 + 6 段任务家族示范）：**6/18 胜、406 步、无效 9.1%**；
  自写 `correct` 提示词 3/18、492 步、9.8%；ReAct 示范（格式转换后）3/18、
  470 步、6.0%；本项目原 `v1` 提示词 **0/18**、540 步、6.9%。
  四版无效动作率都在 6–10%，差别不在动作合法性而在是否推进任务；
  `pick_clean_then_place_in_recep` 在所有版本上都是 0/9。
  **决定**：把 AgentBoard 形式冻结为学生提示词 P_S；因其仓库没有独立 LICENSE
  文件（README 只标 Code Apache-2.0 / Data GPL-2.0），**不复制文本进本仓库**，
  继续以路径 + 仓库提交 + 文件哈希引用。
- **R04｜实现边界（必须随结果一起报告）。** 外部提示词均以 `--prompt-json`
  引入；各家输出格式差异由解析器归一化（`put X in/on Y`→`move X to Y`、
  `clean|heat|cool X using Y`→`... with Y`）。**尚未实现** ReAct 原生的
  "单行输出、`think:` 行不消耗环境步" 语义，也未实现 Agent-R1 的
  `<think>`+`env_step` 工具调用协议；因此本次比选是"提示词内容"的对比，
  不是各框架完整协议的对拍。

## 2026-09-21（续）：冻结提示词后的规模阶梯

- **R05｜在冻结的 AgentBoard 形式提示词下，模型规模与 ALFWorld 成功率单调。**
  评测条件：ALFWorld valid_seen 全 140 局、每局最多 30 轮、贪心解码、
  `Task/Observation/Admissible` 用户轮、六段任务家族示范（AgentBoard
  `prompts/VanillaAgent/alfworld_base.json`，仓库提交 `bb7255e`，文件 sha256
  `3d20f49a…`）。结果：

  | 模型 | seen SR | 无效动作 | 备注 |
  |---|---|---|---|
  | Qwen3-1.7B | 5/140 = 3.57% | 288 | 论文同规模 base 为 25.71%，差约 7 倍 |
  | Qwen3-4B-Instruct-2507 | 45/140 = 32.14% | 446 | 本项目 Student |
  | Qwen3-8B | 69/140 = 49.29% | 259 | 论文 8B base 为 61.43% |
  | Qwen3-14B | 84/140 = 60.00% | 379 | 文档指定的冻结 Teacher |
  | Qwen3-32B-AWQ | 96/140 = 68.57% | 404 | 论文 32B base 为 57.14%，我们更高 |

  结论：**换成外部验证过的提示词后，本项目第一次得到单调、可用的评测阶梯**；
  但仍无法与论文逐点对齐（1.7B 差 7 倍、8B 偏低、32B 偏高），因此跨论文的
  绝对数字不可直接比较，只能在同一协议内部做对照。各模型的短板都集中在
  `pick_clean_then_place_in_recep`、`pick_heat_then_place_in_recep` 与
  `pick_two_obj_and_place`。
- **R06｜对照实验的四要素已固定，可用于四条臂。** Student 初始化 =
  Qwen3-4B-Instruct-2507（base，32.14%）；Teacher = 冻结 Qwen3-14B（60.00%）；
  提示词 = 上述冻结形式；评测 = valid_seen 140 局、30 轮、贪心。四条臂为：
  无更新基线、传统逐 Token OPD、随机状态动作校正、所提选样策略。


## 2026-09-22/23：传统逐 Token OPD 首次长跑与 invalid 输出边界

- **输入（非确认性 Pilot）**：学生轨迹池 `omniopd_runs/student_pool_20260921_02/`
  （240 局、89 胜 = 37.1%、5524 回合、六类各 40 局）→ OPD 训练行
  `omniopd_runs/opd_train_20260922_01/rows.jsonl`（5524 行 + `rows.manifest.json`，
  prompt token 中位 5837、p99 7921）。这些只用于把链路跑通，不是正式确认数据。
- **启动器**：`scripts/run_verl_opd_train.sh`，4 卡（学生 0/1、教师 2/3，Teacher TP=2）、
  `TRAIN_BSZ=64`、`MICRO_BSZ=4`、`LR=1e-6`、`TOTAL_TRAINING_STEPS=86`、K2、温度 1.0、
  `OMNIOPD_REQUIRE_ADMISSIBLE_ACTION=0`。单步约 133 s。
- **E14｜学生输出没有 `Action:` 行时，旧实现直接终止整次训练。** 启动 08 与 14 均在
  step 9 崩溃：`ValueError: Student response contains no action line: 'Task completed:
  cooled some lettuce ... The task cannot be completed as written.<|im_end|>'`。
  宽松分支（`require_admissible=False`）对无 Action 行的输出没有定义监督区间。
  崩溃前的 576 条真实 rollout 中没有一条是这种情况，即**约 1/600 的低频退化生成**
  （模型自述"任务无法完成"）足以让 86 步训练整批作废（单次损失约 20 分钟算力）。
  **处理**：新增纯函数 `emitted_response_span_ids()` 与 `describe_supervision_span()`
  （`src/omniopd/opd_adapter.py`）。宽松分支在无 Action 行时监督**该回合实际生成的全部
  token**：先去掉 veRL 右侧 padding，再剔除末尾终止符；若整条输出只有终止符，则保留这
  1 个 token（否则教师侧会拿到空序列）。严格分支（`require_admissible=True`）保持
  fail-closed 不变。`opd_supervision ∈ {action_line, whole_response, empty}` 随 rollout
  写入 `extra_fields` 记账，不静默丢弃、不免费重采样。注意：`extra_fields` 只在训练
  进程内存在，veRL 的 `rollout_data_dir` 转储只写 `input/output/gts/score/step`，因此
  事后只能用转储文本里"没有行首 `Action:`"的输出条数做近似统计，不能就地读出该字段。
  若要精确计数，必须在同一提交里把计数打进日志或指标，这属于下一轮改动。
  **离线验证**：`scripts/verify_opd_layout.py` 扩为 8 个正例（含"学生放弃"、"放弃且无
  终止符"、"只有终止符"、"整条为 padding"）+ 2 个严格分支必抛用例，断言 mask 宽度/和、
  教师行宽度、监督模式、以及"监督区间 = 实际生成 token"；`tests/test_opd_forward_smoke.py`
  增加 3 条同合同回归测试。
  **必须随结果报告的偏差**：该臂在无 Action 行时监督的是学生整条输出，而不是动作跨度。
  频率约 0.2%，但它是本臂的协议选择，不能外推为通用结论；若后续认为"无效回合应当零权重"，
  需要改协议并重跑，不能只在报告里换说法。
- **E15｜启动 01–14 的其余阻塞项（均已修复并写入仓库）。** 依次为：(1) Hydra 输出目录
  只读导致启动即失败；(2) 上下文预算差 1 个 token；(3) `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments` 与 vLLM 显存池冲突导致引擎初始化失败；(4) worker 硬绑 v1 提示词、
  与学生实际提示词不一致；(5) 无效动作策略报错（改为宽松分支 + 记账）；(6) 响应被截断到
  513 > 512；(7) Teacher 分数宽度与 Student 布局不一致（含差 1 行与超宽两种）；(8) 布局
  逻辑散落在 worker 中，已抽成 `response_mask_for_action_span()` 与
  `align_teacher_rows_to_student_layout()` 两个可离线验证的纯函数。
  14 次启动的结局：01/02/03/04/05/06/07/09/11 在第一个优化步之前失败（配置、服务、
  协议或布局），10/12/13 分别跑到 step 8/11/1 后因布局宽度失败，08/14 到 step 9 后
  因 E14 失败。**到 2026-09-23 为止仍没有任何 checkpoint 产生**。
- **E16｜集群 cron 会用占位任务占满全部 8 张卡。** 09-23 观察到 8 个
  `llamafactory/launcher.py /nfsdir/pulushi/task_test/killme.yaml` 进程各占约 53 GiB、
  ~60% 利用率，父容器名为 `grpo8b`（命令 `sleep infinity`），启动时间与 cron 周期一致。
  这**不是**他人的真实训练，但也不是本项目的作业。处理：`source
  /nfsdir/miniconda3/etc/profile.d/conda.sh && conda activate qwen3_medusa_xh_bak &&
  demokill`（`/usr/local/bin/demokill` 只 kill 匹配 `killme` 的进程）。运行前后必须记录
  GPU 占用，且不得据此推断这些卡长期可用。

- **E17｜八卡训练跑完 86 步却没有落盘，整轮权重全丢。** 2026-09-23 第 16 次运行
  （`opd_standard_20260923_16`）在第 86 步正常收尾、全程零异常、86 条 rollout 全部写出，
  但运行目录里没有 `checkpoints/`。启动器收尾检查随即报
  `training finished without the expected checkpoint`，容器非零退出，完成清单没写，
  已挂好的评测链按设计中止。**代价：约 2 小时八卡训练没有任何权重留下，必须整轮重跑。**
  根因（源码级，不是猜测）：启动脚本传 `trainer.save_freq=0`，而 veRL 的保存分支是
  `if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps %
  self.config.trainer.save_freq == 0 or esi_close_to_expiration)`
  （`/data/yangchunyu/ld/verl-v0.8.0/verl/trainer/ppo/ray_trainer.py:1663`）——
  `save_freq > 0` 是**必需前提**，取 0 时整段保存代码永不进入，连最后一步都不存。
  日志侧证据：`Saved model` 出现 0 次，`save_freq: 0` 出现在解析后的配置里。
  处理：`SAVE_FREQ` 默认 43（86 步运行在第 43 与第 86 步各存一次），启动前校验必须为
  正整数、否则在起容器之前就退出，日志打印实际保存计划，收尾错误信息带 `save_freq`。
  **先行验证**：另跑一次 2 步短跑 `opd_savecheck_20260923_17`，确认
  `checkpoints/global_step_2`（`model_world_size_2_rank_{0,1}.pt`、`optim_*.pt`、
  `extra_state_*.pt`、`lora_train_meta.json`、`huggingface/`）与
  `completion_manifest.json` 都按预期产出（18 个文件、约 18 GiB），然后才启动
  正式的第 18 次运行。该短跑同时确认了 checkpoint 目录结构与
  `scripts/convert_opd_checkpoint.py` 的输入假设一致。
  同一轮还修掉一个同类的启动期配置错误：通用主机侧脚本把 `OPD_DATA` 传成了宿主路径，
  容器内看不到 `/data/...`，现在按容器内 `/runs/...` 寻址并在起容器前校验宿主文件存在。
  **教训**：凡是"跑完之后才有产物"的断言（落盘、完成清单、转换产物），都应当在启动长跑
  之前用一次短跑验证；把失败从"两小时后"提前到"五分钟内"。

- **E18｜checkpoint 里的 LoRA 是 DTensor 分片，旧转换脚本必然失败（同一类"跑完才暴露"的问题）。**
  用第 17 次那轮的 2 步 checkpoint 真跑转换，直接得到
  `RuntimeError: Attempted to access the data pointer on an invalid python storage`
  （safetensors 在 `_find_shared_tensors` 内取 storage 指针时炸）。逐键定位：LoRA 张量
  的类型是 `torch.distributed.tensor.DTensor`，`data_ptr()==0`，
  `placements=(Shard(dim=0),)`、mesh 大小 2、local shard 形状 `(8,2560)`——
  即单个 rank 文件只有一半数据，而且外层包装没有可序列化的 storage。
  旧脚本假设"一个文件里是完整张量"，只做键名映射后交给 safetensors，因此**任何**真实
  checkpoint 都会失败；如果没有这次提前验证，它会在两小时训练结束后才炸。
  **修复**：重写 `scripts/convert_opd_checkpoint.py`——分片按 `(world_size, rank)` 数值排序
  （避免 rank 10 排在 rank 2 前面）；每个 A/B 键取 `to_local()` 后按 placement 维度拼接；
  用分片自述的全局形状与实际重建形状对账；要求每个模块 A/B 成对且 rank 一致；
  传入 `--base-model` 时用 safetensors header 读取基座层形状，逐模块核对
  `A(r, in_features)`、`B(out_features, r)`；输出 `conversion_manifest.json` 记录分片、
  世界大小与全部形状。另新增 `scripts/merge_adapter_for_serving.py`：在 CPU 上用 float32
  计算 base / adapter-applied / merged-reloaded 三组 logits，**adapter 若零效果就拒绝导出**，
  并报出合并误差占 logits 量级的比例。
  **真实验证**（全部在 2 步 checkpoint 上执行）：转换得到 504 个张量、252 个模块
  （36 层 × 7 个投影），与基座形状逐一核对通过；合并后 `adapter_effect=2.6e-3`（非零，
  说明 adapter 确实作用到模型上）、`merge_gap=0.141`、占 logits 量级 0.93%（阈值 25%）→ PASS。
  另加 5 条离线测试（数值排序、键名映射、placement、拼接形状、基座键映射），容器内
  测试套件 **192 项全绿**。
- **E19｜把"跑完才有产物"的检查前移成强制步骤。** 本轮两次损失（E17 无落盘、E18 转换不可用）
  都属于同一模式：断言写在收尾，验算写在事后。新增
  `scripts/host/preflight_opd_chain.sh`：在任何长跑之前先用 2 步短跑把
  "checkpoint → 完成清单 → adapter 转换（含基座对账）→ 合并（含等价性）"整条后处理链
  跑一遍，任何一步失败即在几分钟内暴露。第 18 次运行中第 43 步的中途 checkpoint
  已确认落盘（`checkpoints/global_step_43`，`Saved model` 计数 3），说明落盘修复在
  真实运行里生效。
