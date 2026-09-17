# Agent OmniOPD 实验落地问题记录（截至 2026-09-17）

本文件记录本地代码迁移、服务器环境搭建和传统逐 Token OPD 技术烟测中**已经观察到**的
问题，以及尚未关闭的实验设计关口。它不是实验结果报告，也不表示问题清单已经穷尽。
原 Word 代码的逐项数学与实现缺陷见 [AUDIT.md](AUDIT.md)；新版 veRL 接口及烟测细节见
[VERL_V080_MIGRATION.md](VERL_V080_MIGRATION.md)。服务器 JSON 产物的路径仅作证据索引，
它们尚未随本仓库一同归档。

状态含义：**已处理**＝对应技术问题有修复与有限核验；**待验收**＝已有实现但未通过完整
目标路径；**待决策**＝需要先冻结实验口径；**持续风险**＝不能由一次烟测排除。

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
