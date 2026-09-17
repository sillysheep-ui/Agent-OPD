# veRL v0.8.0 迁移与传统 OPD 对照边界

## 版本和隔离

- 官方发布标签：`v0.8.0`；提交：`7aed6b230776f963fa09509c10d9c3a767d1102c`。
- 该标签源码的 Python 包版本字符串是 `0.8.0.dev`，不是 `0.8.0`；启动器同时核验版本、标签、提交。
- 旧 veRL 0.4.1 checkout、旧镜像和历史产物保留。新版 checkout 应单独放在
  `/data/yangchunyu/ld/verl-v0.8.0`，构建 `docker/Dockerfile.verl080` 的新镜像。
- 仓库自有 `integrations/verl/fsdp_sft_trainer.py` 继续执行行动模仿的加权 CE，
  其 Hydra 配置迁到仓库自有 `configs/verl_v080_omniopd_sft.yaml`。不能把这个配置
  传给 veRL 官方 `main_ppo`，也不能把此训练器称为逐 Token OPD。

## 理论映射

令 Student 在 ALFWorld 状态 (s_t) 下产生动作 Token 序列
(y=(y_1,\ldots,y_L))。传统逐 Token OPD 在 Student 自己的前缀
((s_t,y_{<j})) 上读取 Teacher 的 Token 分布或对应 Token 的 logprob，
对每个有效位置构造 KL/其单样本估计并更新 Student。Teacher 应是被冻结的
本地 Qwen3-14B；Student、Teacher 的 Tokenizer、chat template、思考模式和
响应掩码都要事先核验。

现有 OmniOPD 方法是在 Student 到达的状态上让 Teacher **重新采样动作**，
从有效动作构造监督数据并对 Teacher 动作做加权 CE；它虽然使用 Token CE，
却不是在 Student 的每个输出 Token 前缀查询 Teacher。因此两者的监督单位和
Teacher 预算都不同，不应声称“各 150 次请求”等价于同等计算成本。

## 迁移验收顺序

1. 构建新版镜像并运行 `pip check`、核心依赖与 veRL OPD 模块导入检查。
2. 在新版 checkout 上运行全仓离线测试、静态检查和一个可加载/可保存的
   多卡合成输入 SFT 单步训练；检查梯度、最终 step、LoRA checkpoint 和日志。
3. 再实现 ALFWorld prompt/Student rollout 到 veRL OPD 数据结构的适配，
   明确 action-only response mask、停止条件、环境轨迹和 Teacher token 打分语义。
4. 对传统 OPD 路径做一条真实状态的无更新前向检查，再做单步反向/保存检查；
   记录 Student/Teacher/Tokenizer 的内容指纹与每 Token/每状态 Teacher 成本。
5. 用同一 SFT 初始 Student、相同训练/测试游戏划分及评测协议，分别重跑
   无更新基线、传统逐 Token OPD、随机状态动作校正、所提选样策略；
   预先定义预算匹配方式和超参搜索预算，避免用旧清单混用新版实验。

完成前两项只表示**旧行动模仿链路在新版环境通过技术验收**；第三、四项完成
之前，不得宣称传统 OPD 对照可运行。任何正式结果必须用新代码 revision
重新生成上游清单，旧版 `code_revision`/code-tree 哈希不得复用。

## 2026-09-17 已完成的技术烟测

- 服务器上独立 checkout：OmniOPD `d35ce4768aa72e2e9fc40562e6395fccd9853572`
  与 veRL `7aed6b230776f963fa09509c10d9c3a767d1102c`，工作树干净。
- 新镜像 `omniopd-verl080:alfworld` 的 ID 为
  `sha256:1a13f969aa56412e0a73f85f1b334caee07b52cdb8d2cbb71285ed15f36c3ebb`；
  继承旧镜像
  `sha256:55257bca4c7d02a3a5f42833b62f3edd595048b413826c64b4c7393080dbcdf6`。
  只升级 TensorDict `0.6.2→0.8.3`，`pip check` 无断裂依赖。
- veRL `0.8.0.dev`、`verl.trainer.distillation.losses` 与 `main_ppo` 可导入；
  本仓 149 项离线测试和 Ruff 静态检查通过；Hydra 自有配置可解析，
  自有 FSDP trainer 在新版 veRL 上可导入。
- 使用旧预检中的两条**合成** JSONL 和既有 Qwen3-4B Student，
  GPU 0–3 完成一次 FSDP1/LoRA 训练、完整验证和 `global_step_1` 保存：
  train loss `1.1026837682948099e-06`，val loss `6.854526191091281e-07`，
  grad norm `8.722222992219031e-05`。结构校验确认 PEFT LoRA rank 16、
  alpha 32、252 对 A/B tensor；GPU 0 上 base+LoRA 实际加载成功。
  输出保存在 `/data/yangchunyu/ld/omniopd_runs/verl080_migration_smoke_20260917`。
  训练后记录的 LR 为 0，是 1-step cosine scheduler **下一步**的学习率，
  不表示本步梯度为零。

这些结果不包含 Qwen3-14B Teacher、Teacher Token 分布、ALFWorld 正式数据、
正式 launch/completion manifest 或环境交互对照；它们不验证论文结论。
第 3、4、5 项仍未完成，当前**不能启动传统 OPD 确认性实验**。

## 逐 Token OPD 适配进度与未闭合接口

- `src/omniopd/opd_adapter.py` 已能把固定状态池中的 Student 历史转换为
  Student 提示词，并另存由 Teacher 系统提示词构造的评分上下文；不会把旧
  Student 行为动作或 Teacher 新采样动作误当成 OPD 目标。
- `scripts/build_opd_prompts.py` 只针对已登记的均匀嵌套选样生成提示词行，
  逐项核对状态池、选样清单、实验配置、文件哈希和当前代码版本。输出清单明确
  标记 `training_ready: false`。代码改动后须重新生成状态池与选样文件，不能
  使用旧版确认性实验的清单。
- `scripts/audit_opd_tokenizers.py` 在不加载模型权重的前提下，核对 Qwen3
  Student/Teacher 的 Token ID 空间、编码规则、特殊 Token 和非思考模式提示词；
  它只证明 Token ID 可比较，不证明 Teacher 权重、服务或打分接口正确。
  本次服务器预检发现 SFT Student 的模板在 assistant 头后直接生成动作，而
  Qwen3-14B Teacher 的模板在 `enable_thinking=False` 时先插入一个已闭合的
  空 `<think>…</think>` 段。两模板不能要求文本完全相同：正确的条件是共同
  Token ID 语义、两侧提示词各自正确编码、Teacher 在其独立前缀下给同一批
  Student 动作 Token 评分。审计会记录两份模板哈希及实际生成前缀。
- `align_teacher_sampled_token_logprobs` 为 Student 生成的 Token 对齐 Teacher
  的逐位置 logprob，并拒绝缺失、非有限值和序列错位。尚未与 veRL worker
  的实际返回结构连接，也未通过真实模型端到端测试。

下一道实现关口是：从 Student **实际采样**的动作构造 action-only 掩码，按
`P_T(s)` 加同一组 Student Token ID 请求冻结 Teacher 的逐位置评分，再将
对齐后的有效 Token logprob 接入 veRL 的 OPD 损失。官方 veRL 当前默认的
Teacher 打分把 Student 提示词和响应 Token 一起传入，不能满足这里两个系统
提示词不同的定义；必须使用本仓自有的评分适配。固定状态池只代表条件于
所选状态的 Token-on-policy 更新，不代表训练中实时交互形成的新状态分布。

`scripts/smoke_opd_teacher_scores.py` 是单条真实状态的**无更新技术烟测**：
现场采样 Student 动作、拒绝非单行或不可执行的响应、排除 EOS 并用 Teacher
独立提示词对同一动作 Token 逐位置评分。旧状态池可供这一烟测，但不能产生
确认性结果。该脚本的 Hugging Face 前向分数仍需与后续 veRL worker 的实际
分数逐 Token 对拍；它不替代 veRL 损失接线、梯度检查、训练清单或预算审计。
