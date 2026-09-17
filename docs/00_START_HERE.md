# Agent OmniOPD 实验执行交接入口

本文件是人类实验者或具备终端能力的执行模型进入本代码库时的第一入口。先阅读本文件，
再阅读 `README.md`、`docs/REPRODUCTION.md` 和 `docs/VERIFICATION_REPORT.md`。

> **版本迁移提示：** 当前工作分支正在从 veRL 0.4.1 迁到 v0.8.0。
> 下文所述“可执行”是旧版已验收的行动模仿主链，不代表新版多卡验收
> 或传统逐 Token OPD 对照已经完成。新版门槛见 `docs/VERL_V080_MIGRATION.md`。

本地/服务器落地过程中已观察到的问题、处理状态、跨实验复用检查及后续追加规则见
持续维护的 `docs/EXPERIMENT_ISSUES_20260917.md`；不要把单状态技术烟测当作正式结果。

## 1. 当前可以执行的实验

当前规范生产链完整覆盖同一 Student state pool 上的 fixed-budget 对照：

- breadth：\(G=50,m=3,M=150,N=1,B=150\)；
- depth：\(G=50,m=1,M=50,N=3,B=150\)；
- 四个训练种子：`7, 42, 123, 2024`；
- 每个训练臂固定 `102` 个 optimizer steps。

state-source control 配置是不可运行的设计约束模板。M3/SAGE 分析器要求的 A1/A3
\(N=1\) 上游 pair/training 生产链及 M3 多训练种子最终汇总尚未实现，不能把分析入口
存在解释为这些确认性实验已经可运行。

## 2. 压缩包已经包含

- `src/omniopd/`：规范实现；
- `scripts/`：state pool、选择、Teacher标注、数据、训练、评测和分析入口；
- `configs/`：预注册协议和固定预算配置；
- `docker/Dockerfile.verl041`：锁定 veRL/ALFWorld/TextWorld 运行依赖的目标集群镜像；
- `docker/Dockerfile.verl080`：隔离的新版候选镜像定义；
- `integrations/verl/`：本仓库自有的行动模仿训练器，不是传统 OPD 训练器；
- `tests/`：离线协议与对抗式回归测试；
- `docs/REPRODUCTION.md`：完整命令顺序；
- `docs/EXECUTION_INPUTS.template.yaml`：外部资源填写模板；
- `docs/EXECUTOR_PROMPT.md`：可直接提供给另一个执行模型的任务说明；
- `docs/CODE_INVENTORY.json`：逐文件字节数和 SHA256；
- `docs/ALFWORLD_REBUILD_20260916.md`：官方 TextWorld 数据重建与目标服务器验收记录；
- `legacy/`：两份 Word 中旧代码的只读审计快照，禁止作为运行时代码。

## 3. 无法安全内嵌、必须由实验者提供的资源

- 可完整加载的 behavior Student 模型与同一 Tokenizer；
- ALFWorld 数据、TextWorld games 和环境配置；
- 旧版复现用 veRL 0.4.1；新版迁移用单独的 v0.8.0 checkout；
- CUDA/GPU、可用的本机回环端口和输出存储目录；
- Teacher 模型名、provider revision、Tokenizer与context window；
- 通过环境变量 `DEEPSEEK_API_KEY` 注入的密钥。

不要把真实密钥写入模板、命令行、日志、Git或manifest。

## 4. 首次解压后的强制步骤

交付 ZIP 不包含 `.git`。在解压后的 `Agent_OmniOPD` 目录执行：

```bash
git init
git config user.name "<operator>"
git config user.email "<operator-email>"
git add -A
git commit -m "freeze Agent OmniOPD experiment release"
git status --short
```

最后一条命令必须无输出。生产入口会绑定当前 Git revision 与 canonical code-tree hash；
开始实验后修改 `src/`、`scripts/`、`integrations/`、`configs/`、`pyproject.toml` 或
`README.md` 会导致后续阶段故意拒绝旧清单。

## 5. 执行顺序

1. 复制 `docs/EXECUTION_INPUTS.template.yaml` 到仓库外部，填写全部 `REQUIRED_*` 字段；
2. 安装依赖，运行两套测试、静态检查、Python编译与两组配置验证；
3. 检查模型、Tokenizer、环境、veRL revision、GPU、端口和输出目录；
4. 使用新目录运行探索性 smoke test，验证真实 ALFWorld、vLLM、Teacher API 与 LoRA
   保存/加载；不得把 smoke 数据混入确认性结果；
5. 向负责人报告 smoke test 的命令、清单、API调用数、日志和失败项，取得正式运行授权；
6. 严格按 `docs/REPRODUCTION.md` 从新的 immutable Student state pool 开始执行正式链；
7. 每一阶段只消费前一阶段通过内容哈希和语义验证的实体，不按文件名猜测身份；
8. 保存所有 manifest、request ledger、resolved config、日志、checkpoint 和逐 seed/game
   汇总；任何阶段失败都换新输出目录，不删除 invalid/error attempt 后补打免费请求。

## 6. 开始付费或长时间运行前的门槛

执行模型必须先报告以下项目，未经负责人确认不得启动正式 API/GPU 作业：

- 代码工作树干净，离线测试和配置验证全部通过；
- 所有 `REQUIRED_*` 外部输入已解析为存在的明确路径或稳定标识；
- behavior Student/Tokenizer 内容身份一致；
- ALFWorld/TextWorld 运行时与逐 game seed 能被记录；
- 按所选代码版本核验 veRL 的标签、提交与干净工作树；当前启动器要求 v0.8.0；
- 预算为 \(B=MN=150\)，两臂共享 Teacher 协议和 annotation pair；
- 预计 API 调用量、GPU 数量、运行时长与输出位置已经得到授权。

## 7. 结果解释边界

离线测试通过只证明代码满足已写明的协议，并不证明论文经验结论。正式报告必须明确区分：

- Teacher validity 条件后的 \(q_T^V(a\mid s)\) 与原始 \(q_T\)；
- 模型实际上下文 \(\tau_\Lambda(z_t)\) 与用于replay的完整历史；
- \(N\)-dependent retention 与纯 Monte Carlo 方差；
- M1 surrogate alignment、M2 marginal representativeness、M3 action-imitation transfer、
  SAGE intervention labels 和 Position consequentiality；
- 代码验证、目标环境集成验证与最终经验结论复现。
