# Agent OmniOPD 本地执行、Git 管理与 GitHub 交付需求文档

版本：1.0  
日期：2026-09-11  
适用仓库：`agent_omniopd`  
当前基线提交：`7acd71b`（`refresh handoff package inventory`）

## 1. 文档目的

本文件用于把 Agent OmniOPD 从“已有代码和阶段性结果”整理为一个可以在本地或 GPU
服务器上安装、审计、执行并通过 GitHub 交付的工程。它规定：

1. 当前仍缺失的代码、环境、数据和实验材料；
2. 本地环境与 Git/GitHub 的管理方式；
3. 哪些内容可以上传 GitHub，哪些内容必须留在仓库外；
4. 正式实验开始前、运行中和结束后的验收标准；
5. 执行人员需要向负责人补充的信息和授权。

本文不授权启动付费 Teacher API、GPU 长任务或上传仓库。

## 2. 当前状态

### 2.1 已具备

- `src/omniopd/` 中已有规范实现；
- breadth `150×1` 与 depth `50×3` 的固定预算主链已有配置和运行入口；
- state pool、selection、Teacher annotation、数据构建、LoRA 训练、评测、聚合和分层
  bootstrap 已有脚本；
- Position 反事实分析已有规范入口；
- 已有离线测试、理论—代码映射、审计报告和复现手册；
- 当前仓库已初始化 Git，分支为 `main`，检查时基线提交为 `7acd71b`；
- 检查时基线工作树干净，尚未配置 GitHub remote；
- `data/`、`outputs/`、`checkpoints/`、日志和虚拟环境已被基础 `.gitignore` 排除。

### 2.2 当前尚不能声称

- 不能声称任意新机器克隆后即可一键复现；
- 不能声称原定 4-GPU 确认性协议已经完成：已有主实验实际为 2-GPU 协议修订实验；
- 不能把 Base 对照写成预注册确认性主结论；它是经审计的事后次级诊断；
- 不能声称 Position 已完成，也不能用 breadth Position 代表 depth；
- 不能声称 A1/A3、M3、SAGE 的确认性生产链已经闭环；仓库目前主要完整支持
  fixed-budget breadth/depth 主链。

## 3. 当下仍缺失的内容

### 3.1 必需的外部执行资源

下列资源不应硬编码或直接提交到 GitHub，但本地执行前必须准备：

- 完整、可加载的 behavior Student 模型目录及完全一致的 Tokenizer；
- 模型和 Tokenizer 的内容哈希、来源、版本或不可变 revision；
- ALFWorld 数据、环境配置、game 文件以及准确版本；
- TextWorld 版本和环境随机种子控制能力；
- 干净的 veRL `0.4.1` checkout、其 Git commit 和依赖补丁说明；
- 兼容的 NVIDIA GPU、驱动、CUDA、PyTorch、vLLM 和显存配置；
- Teacher provider 的 base URL、模型名、模型 revision、Tokenizer 身份和 context window；
- 通过环境变量注入的 `DEEPSEEK_API_KEY`；
- 足够的 checkpoint、日志、评测和归档存储空间；
- 可用的本地回环端口及相应的进程管理权限。

### 3.2 尚未随当前代码仓库交付的正式产物

当前 Git 仓库主要是代码库。要让第三方复核既有实验，还需要从原执行机器单独归档并提供：

- 原始 state pool、selection、corrections 和完整 request ledger；
- annotation pair manifest；
- 8 个训练任务的 launch/completion manifest、训练日志和最终 LoRA checkpoint；
- 冻结的 134-game 列表、逐 seed×game 评测明细和服务证明；
- `audit_report.json`、`audit_report_v2.json`、checkpoint 指纹检查结果；
- Base、NLL、任务分层、功效分析等结果的原始输入、脚本版本和完整输出；
- 所有文件的完整 SHA256，不使用缩写哈希替代归档身份；
- 既有运行所使用的实际环境清单，而不是计划模板。

这些产物若含模型权重、受许可限制的数据、私有路径或服务信息，不上传普通 Git 历史；应放入
受控对象存储、私有 Release、Git LFS 或 DVC，并在 Git 中只保存脱敏 manifest 和校验值。

### 3.3 统计与报告待修正项

正式论文报告前至少要完成：

1. 解释并复核 NLL 的口径。`3.94 → 2.06–2.17` 与“配对差异 `−1.06～−1.41`”
   算术上不一致；如果分别使用 token 加权和样本/游戏等权，必须分别命名估计量。
2. NLL 的不确定性必须按 game 聚类，而不是把同一游戏中的 14 个 state 当作独立样本；
   若仅有约 5 个 game，应将其定位为探索性诊断。
3. 将主结果表述为“经审计的、包含已记录协议修订的主分析”，不写成严格 4-GPU 确认性结果。
4. 将 Base 比较标记为“经审计的事后次级诊断”。
5. 修正“简单/清洁任务不受损或提升”的概括，并为所有任务表明确分母、arm、seed 和
   loop 定义。
6. 在最终报告中加入 canonical code-tree hash、完整 Git commit、脚本 SHA256、输入输出
   SHA256、checkpoint tree hash、环境版本和服务指纹。

### 3.4 尚待执行的机制实验

- 先做 5–10 个预先指定 breadth state 的 Position 技术试运行；
- 试运行只验证 replay fidelity、环境恢复、请求计数、运行时间和同动作对称性，不做推断；
- 通过后在全新目录运行全部 145 个有效 breadth state；
- 如需解释两臂差异，再运行 depth Position，并保留每个 state 的全部有效 Teacher draws；
- 可选：补做 4-GPU 敏感性实验；它不能追溯性替代原2-GPU主实验；
- 可选：检查中间 checkpoint、更新尺度和训练动态，用于解释“Teacher动作有利但LoRA变差”
  等情形。

## 4. 本地环境管理要求

### 4.1 管理原则

Git 负责管理代码、配置、环境声明、迁移记录和小型脱敏结果，但 Git 本身不能冻结 GPU
驱动、CUDA 内核、系统库或外部模型。建议采用以下组合：

- Git + GitHub：代码、配置、文档、CI 和版本标签；
- `uv` + `uv.lock`：Python 依赖锁定；若执行环境不允许使用 `uv`，使用明确版本的
  `requirements/*.txt` 锁文件；
- `environment.snapshot.json`：记录操作系统、Python、CUDA、驱动、GPU、PyTorch、
  Transformers、vLLM、ALFWorld、TextWorld、veRL revision；
- Git LFS、DVC 或对象存储：大文件、模型、checkpoint 和正式运行产物；
- 环境变量或本地 secret manager：API 密钥。

不得仅以 `pyproject.toml` 中的下限版本作为复现实验环境。当前的 `numpy>=...`、
`transformers>=...` 等范围可能在未来解析出不同依赖集合。

### 4.2 建议的平台拆分

由于完整实验依赖 CUDA，推荐明确分成两类环境：

- 开发/审计环境：macOS 或 Linux，可运行离线测试、格式检查、统计分析和文档生成；
- 正式 GPU 环境：Linux + NVIDIA GPU，运行 vLLM、veRL、训练、评测和 Position。

macOS 本地环境不能作为 CUDA 训练环境的等价替代。

### 4.3 首次安装验收

在新克隆仓库中必须做到：

1. 从锁文件创建独立虚拟环境；
2. 安装项目及 test/api/train 依赖；
3. 运行全部离线测试；
4. 运行静态检查和 Python 编译检查；
5. 验证 breadth/depth 配置对；
6. 生成环境快照；
7. 确认 Git 工作树干净；
8. 在未授权前不启动 Teacher API 和长时间 GPU 任务。

当前机器直接检查时缺少 `PyYAML` 和 `ruff`，因此测试与静态检查不能在系统 Python 中
直接通过。这属于环境未建立，不是已经证明的代码失败；必须在锁定虚拟环境中重新验收。

## 5. Git 工作流要求

### 5.1 仓库边界

GitHub 仓库根目录必须是 `agent_omniopd/`。不得把其外层目录中的简历、Word 文件、临时
渲染文件和多个历史 ZIP 一并提交。

### 5.2 分支与提交

- `main`：只保留已验收、可复现的代码；
- 功能分支：`feat/<name>`（仓库既有分支即用此前缀，例如 `feat/verl-v080-opd`）；
- 修复分支：`fix/<name>`；
- 实验协议变更：`protocol/<name>`；
- 一个任务只占一个专用分支；同一任务的主题拆分用同一分支内的多个提交表达，
  确需独立分支时从该任务分支派生，不平行创建多个同源分支；
- 每个提交只承担一种清晰变更；
- 合并前必须通过测试、静态检查、配置验证和敏感信息扫描；
- 正式实验从带签名或受保护的 Git tag 启动，例如 `omniopd-v1.0.0`；
- manifest 同时记录 Git commit、Git tag 和 canonical code-tree hash；
- 正式运行期间不直接修改代码。若必须修复，创建新提交、新 tag 和新输出根目录。

### 5.3 `.gitignore` 和秘密管理

上传前应补充排除：

- `.env`、`.env.*`、`secrets/`、凭据文件；
- 本地执行输入文件，例如已填写的 `execution_inputs.local.yaml`；
- IDE、操作系统临时文件；
- 本地服务 PID、socket 和运行时缓存；
- 任何未脱敏的 API request/response dump。

模板只能保留 `REQUIRED_*` 占位值。真实 API key 不得进入命令行、Git、日志或 manifest。

### 5.4 GitHub 自动验收

至少建立一个不依赖 GPU 和真实 API 的 CI：

- 支持声明的最低 Python 版本以及正式使用的 Python 版本；
- 安装锁定依赖；
- 运行离线测试；
- 运行 Ruff 和编译检查；
- 验证两臂配置；
- 检查文档中的内部链接；
- 运行 secret scan；
- 检查代码清单或 canonical hash 是否需要更新。

当前已实现：`.github/workflows/ci.yml` 在 push 与 PR 上运行 `compileall`、
`scripts/run_tests.py` 与 Ruff，矩阵覆盖 Python 3.10/3.11/3.12。secret scan、配置
一致性检查和 canonical hash 新鲜度检查仍未接入 CI，需继续补齐。

GPU、vLLM、ALFWorld 和 Teacher API 集成测试默认不在公共 CI 中运行，可在自托管 runner
或人工 smoke gate 中执行。

## 6. GitHub 上传要求

上传前负责人必须确定：

- GitHub 组织/账号和仓库名称；
- 公有还是私有；
- 是否保留当前本地 Git 历史；
- 代码许可证；
- ALFWorld、模型、Teacher输出和 veRL 派生代码的再分发边界；
- 是否使用 Git LFS/DVC，以及大文件实际存储位置；
- 哪些实验结果允许公开；
- 是否启用 branch protection、required checks 和 tag/release 规则。

推荐顺序：先建私有仓库并完成秘密扫描、许可证审核和 CI，再决定是否公开。未经上述选择，
不得自行创建公开仓库或推送历史。

## 7. 正式实验执行门槛

只有以下条件全部满足，才允许启动正式任务：

- GitHub 或本地 Git 中存在冻结提交和实验 tag；
- 工作树干净，canonical code-tree hash 已记录；
- 锁定环境安装完成，离线测试、静态检查和配置验证全部通过；
- Student/Tokenizer、ALFWorld/TextWorld、veRL、Teacher身份均已解析；
- 实际 GPU 数量已经定稿，配置、手册和运行命令一致；
- smoke test 使用独立目录且通过，不混入正式结果；
- Teacher调用上限、GPU小时和存储预算获得负责人授权；
- 输出目录为空且不会覆盖旧实验；
- 备份、失败恢复、partial ledger 保留策略已经确定。

## 8. 最小交付物

### 8.1 GitHub 代码仓库

- 源代码、脚本、配置和测试；
- `README.md` 和完整执行手册；
- Python 锁文件；
- CI 配置；
- `LICENSE`、第三方依赖/数据许可说明；
- 加强后的 `.gitignore`；
- 示例配置和环境快照生成工具；
- release tag、变更记录和代码清单。

### 8.2 私有实验归档

- resolved config；
- 全部 manifest 和 ledger；
- 环境快照；
- 训练日志和 checkpoint；
- 逐 game、逐 seed 原始结果；
- 统计脚本输入和输出；
- 完整 SHA256 清单；
- 一份说明“确认性、协议修订、事后、探索性”证据等级的最终报告。

## 9. 验收标准

### 9.1 仓库验收

- 新机器可以仅依据 README 和锁文件建立开发环境；
- 无真实 API、无 GPU 时，离线 CI 全部通过；
- 仓库中不存在密钥、私有绝对路径、模型权重或受限制数据；
- GitHub Release/tag 对应唯一 commit 和代码清单；
- 文档没有把尚未闭环的 A1/A3/M3/SAGE 写成可直接完成的确认性实验。

### 9.2 实验验收

- 所有阶段的输入身份可由哈希反向核验；
- 两臂共用同一 state pool 和 Teacher contract；
- invalid/error 请求完整计入预算；
- 训练步数、数据、Student、Tokenizer和评测游戏一致；
- 主统计同时传播 training-seed 和 game 不确定性；
- Position 使用按 game 聚类的推断并明确其单步干预解释边界；
- 报告中的每个数字均能由归档原始输入重新生成。

## 10. 当前执行困难与风险

1. **正式实验资产不在当前仓库内。** 当前无法仅凭代码重新验证既有 checkpoint、原始
   ledger、134-game结果和全部审计 JSON。
2. **环境尚未锁定。** 当前只有依赖范围，没有 lock file；系统 Python 缺少必要依赖，
   不能代表目标环境测试结果。
3. **本机不等于目标 GPU 服务器。** CUDA、vLLM、veRL 和 ALFWorld 的真实集成必须在
   Linux GPU 环境验证。
4. **2-GPU与4-GPU口径尚需负责人决定。** 模板仍写4 GPU，而既有主实验实际为2 GPU；
   新实验必须明确是复现既有修订协议还是运行4-GPU敏感性实验。
5. **GitHub目标尚未提供。** 当前没有 remote，也没有仓库名称、可见性、账号/组织和认证；
   因此不能安全推送。
6. **许可证尚不完整。** `pyproject.toml` 只有研究用途提示，没有独立 `LICENSE`，且模型、
   ALFWorld数据、Teacher输出、veRL相关文件的再分发权需要单独核对。
7. **大文件策略尚未确定。** checkpoint、模型和正式结果不适合普通 Git；需要确定
   LFS、DVC或对象存储。
8. **统计报告仍有待修正。** 尤其是NLL口径和聚类推断，修正前不应成为论文定稿证据。

## 11. 负责人需要提供或决定的事项

- behavior Student/Tokenizer 的位置、来源和是否允许复制；
- ALFWorld/TextWorld数据及目标 split；
- GPU服务器规格和可用GPU数量；
- 本次新执行采用2 GPU还是4 GPU；
- Teacher provider信息、调用预算和密钥注入方式；
- 既有正式实验归档的位置；
- GitHub账号/组织、仓库名、公有或私有；
- 代码和实验材料的许可证；
- 大文件与敏感结果的存储方案；
- 是否只完成代码工程化交付，还是继续完成 Position、4-GPU敏感性和训练动态诊断。

## 12. 建议实施顺序

1. 收集并只读归档既有实验产物，生成总 SHA256 清单；
2. 决定2-GPU/4-GPU协议和GitHub公开边界；
3. 补充锁文件、`.gitignore`、许可证、CI和环境快照工具；
4. 在功能分支完成修改并通过离线验收；
5. 建立私有GitHub仓库，推送代码并启用分支保护；
6. 在Linux GPU环境执行独立smoke test；
7. 修正NLL和报告措辞；
8. 经负责人授权后，再运行Position或其他新增实验；
9. 生成代码Release与独立实验归档，不把大文件和密钥写入Git历史。

