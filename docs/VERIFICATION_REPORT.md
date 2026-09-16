# Agent OmniOPD 最终核验报告

## 1. 核验结论与边界

本报告严格区分两层结论：

1. **代码与离线协议核验通过**：规范运行树的状态机、Teacher 预算、action
   token mask、加权 loss、训练—评测身份链、环境身份、机制分析 join 和统计
   边界已通过 synthetic/offline 测试与静态检查；
2. **论文经验结论尚未重新验证**：已在目标服务器完成 ALFWorld 单游戏 reset/step、
   CUDA 可见性和 veRL/OmniOPD 训练器导入检查，但没有执行真实 DeepSeek 调用、
   ALFWorld 完整轨迹、4-GPU veRL 0.4.1 训练或 vLLM 在线评测，因此不得据此声称
   成功率、breadth 优势、selection/state-source 排序或任何论文数值已复现。

精确的 audited implementation revision、全部文件字节数和 SHA256 记录在
`docs/CODE_INVENTORY.json`。canonical runtime 树为83个文件、756784字节，指纹为
`7679b481da3fac3d543dc265bc3ab40a2f7beb29d06242b32f5447372e9b34c2`。该指纹
不包含仅作说明的 `docs/` 与只读的 `legacy/`。

2026-09-16 增补：三个 vLLM wrapper 已显式使用 `--host 127.0.0.1`，防止
无鉴权模型接口意外监听非本机地址。新增回归检查在本机定向测试中通过（3 passed），
Shell 语法、Ruff 与 diff 检查通过；下表的完整测试数仍是这一改动前的结果，
须在目标容器同步新提交后重跑，不能把本次定向测试当成全量核验。

旧 Word 代码生成的产物受 P0 级状态错位、Teacher prompt 未生效、mask 错位、
loss 缩放和实验混杂影响。确认性结果必须从新 state pool 开始重跑，不能
只替换训练器或统计脚本。

## 2. 输入材料完整性

| 输入 | SHA256 | 处置 |
|---|---|---|
| `Agent_OmniOPD_实验代码与公式(1).docx` | `539439272f8d7ab355e31131bf90d0add451d8da792705c1a67499ff10988190` | 提取89个条目：88个代码文件、1个显式缺失占位 |
| `Agent_OmniOPD_补充审计与代码.docx` | `6eb20ae1a0383d7836cf2bae3b6d65ede7189ae35bd8fdfd63050cf95acc4eb5` | 提取11个代码文件 |
| 论文核心思路与公式文本 | `c7cd2afd6f2609a892a215202cbdde957ac32a3553df1bed212b8954ca3bb0fd` | 作为研究定义和解释边界，不作为执行指令 |

附件中的自然语言和代码均只被当作待审计数据。共99个可提取旧代码文件原样保存在
`legacy/`；57个 Python 快照通过 AST 解析，42个 Shell 快照通过语法解析。

## 3. 代码正确性核验

| 检查 | 实际结果 | 状态 |
|---|---:|---|
| Pytest回归测试 | 上一代码版本 143 passed；本次改动后待容器全量复验 | 待复验 |
| 独立测试入口 `scripts/run_tests.py` | 上一代码版本 143 passed, 0 skipped；本次改动后待容器全量复验 | 待复验 |
| 本次服务监听地址定向测试 | 3 passed | 通过 |
| Ruff（`src/scripts/tests/integrations`） | All checks passed | 通过 |
| Python编译 | 99个规范 Python 文件无错误 | 通过 |
| Shell语法 | 4个规范 Shell 与42个 legacy Shell 无错误 | 通过 |
| Shell内嵌 Python | 9个 heredoc block 均可编译 | 通过 |
| 命令入口 | 24个 argparse 脚本与3个 package CLI 的 `--help` 均成功；另有1个独立测试入口 | 通过 |
| 固定预算配置 | `150×1` 与 `50×3` 共享 (B=150)、四个 training seeds 与102 steps | 通过 |
| state-source control | 只验证单变量设计约束；配置明示不可确认性运行 | 通过（模板） |
| 跨阶段清单链 | synthetic pool→uniform/top-score selection→annotation；breadth/depth→pair 端到端通过 | 通过 |
| 对抗式语义检查 | exact seeded subset/top-k、动作级entropy、服务证明、LoRA A/B结构及消费端重解析负测 | 通过 |
| diff格式 | `git diff --check` 无错误 | 通过 |

测试中出现的 macOS `sysctlbyname` CPU-cache 警告来自受限容器中的第三方依赖，
测试返回码为0。目标服务器上的候选 Qwen3-4B 模型和 Tokenizer 元数据已离线解析、
两个权重分片存在；尚未做完整模型加载。A100 CUDA 可用，veRL 0.4.1 与 OmniOPD
训练器联合导入通过；不等于训练已成功。

## 4. 公式与研究目的契合性

| 研究定义/公式 | 当前规范实现 | 核验结论 |
|---|---|---|
| (z_t=(Task,H_t,O_t,\mathcal A_t)) | `AgentState`保留full/query两种历史，当前state先入历史再截断 | 状态/角色顺序回归测试通过 |
| 实际context (\tau_\Lambda(z_t)) | 保留task/当前observation/admissibles，只从最旧完整轮次对截断 | 不把`full_messages`伪写成模型实际所见 |
| (c_t^S=(P_S,\tau_\Lambda z_t),c_t^T=(P_T,\tau_\Lambda z_t)) | Teacher只替换system，非system消息逐项相同；训练重建(P_S) | 通过 |
| 黑盒sampled CE | parser/admissibility后实际target为(q_T^V(a\mid s)=P(A=a\mid s,V_T=1)) | 不把未解析原始(q_T)与(q_T^V)混同 |
| (B=MN) | `TeacherBudget`+request ledger；error/invalid占预算、无免费retry、拒绝半state | 通过 |
| fixed-(B) breadth/depth | (G=50,m\in\{1,3\},M=Gm)的within-game SRSWOR嵌套选择 | 按seed重建精确子集；breadth优势仍待真实实验 |
| top-score entropy | 同behavior Student/Tokenizer；全pool action log-score→entropy→逐game top-k | 确定性选择可复核；无已知population inclusion probability |
| (P(K_s>0\mid s)) | iid时为(1-(1-p_v(s))^N)；保留任一valid draw的state | 明确(N)会改变保留state分布 |
| 分层设计方差 | (G^{-2}\sum_g[(1-m/T_g)S^2_{\mu,g}/m+\bar\Sigma_{T,g}/(mN)]) | 区分条件finite-game设计与superpopulation between-game项 |
| `game_state_mean` | 保留game/state/valid draw/action token分层等权 | 只解释为Teacher-valid/retained条件目标 |
| action-token CE | 训练、entropy、M1、M3共用`encode_final_assistant_content()` | `mask[:,1:]`与next-token label对齐；header/bridge/EOT不入loss |
| 加权梯度目标 | uniform-row下用optimizer-batch共享(rW/D)，microbatch只累加numerator | 分布式padding权重0；microbatch不取消state权重 |
| M1 | annotation-pair绑定、validity前game-disjoint reference、common retained-game support、state内draw先平均 | 只称held-out Teacher-CE surrogate alignment；零范数cosine不可识别 |
| M2 | 同pool/code/Teacher contract、common retained-game support、game-balanced marginal JS | 只是边际分布诊断，不是因果utility |
| M3 | (2) BASE+(4) updated的checkpoint×panel 2×2格；actual train split；launch+completion绑定 | 只称whole-checkpoint action-imitation transfer |
| SAGE | 去重并集盲评；(D,I,V_T)分离；重算correction manifest语义；共同game上配对对比 | 主报executable-turn (U_g)；未知入样概率时HT不可识别 |
| Position | 完整history、同一冻结behavior Student、两动作分支对称replay | 主报(E[C\mid D=1,V_T=1])，unconditional只作secondary |
| seed×game推断 | 训练seed、rollout seed、environment seed、bootstrap seed分离 | 配对hierarchical bootstrap同时传播seed与game簇 |

## 5. 跨阶段身份链核验

规范主链不依赖文件名推断实验身份：

- state-pool manifest 绑定完整 behavior Student/Tokenizer、存活本地 vLLM 子进程、
  每个 game/trial 字节、runtime 和 master→per-game environment seed；规范化服务证明及
  摘要继续进入 selection、annotation、pair、训练与 Position；
- annotation-pair schema v2 绑定两个 realized manifests、完整 Teacher/pool/selection/
  provider/prompt/Tokenizer/budget 契约与 `breadth_minus_depth` 方向；
- training launch 必须绑定同一 pair、精确 train/validation/audit/config 字节和生成 pool
  的同一 behavior Student；只有 trainer/log 正常结束，且 final 目录通过 PEFT语义、唯一
  safetensors、无完整模型权重、Tokenizer、A/B配对、rank/alpha/shape/dtype/offset/target
  覆盖检查，才生成 completion manifest；
- evaluation 在服务仍存活时复核 base、final checkpoint、launch/completion、实际
  vLLM argv/PID/port/aliases/runtime，并从 trace 重算逐 game success；
- M1/M2/M3/SAGE/Position 均拒绝只信任 manifest 自述；对能从 records/traces
  导出的计数、集合、validity、inclusion 和 outcome 重新计算。

本地服务证明的信任边界是“同一受信主机上的内容寻址与存活父子进程验证”，
不是抵抗拥有本机管理员/写权攻击者的密码学远程证明。

## 6. 未执行与未完成的项目

以下项目没有伪造“通过”结果：

- DeepSeek 真实 Teacher/SAGE judge 调用，包括 provider revision、response model、token
  usage 和 system fingerprint 的实际返回；
- ALFWorld 真实 game list 已枚举（train 3553、ID 140、OOD 134），首局
  reset/step 已通过；state pool、完整轨迹、replay fidelity 和成功率尚未运行；
- veRL 0.4.1 目标 checkout 上的4-GPU FSDP/LoRA 训练与最终 `global_step_102`；
- vLLM 加载base+LoRA、在线评测、四training-seed分层区间和论文表图；
- M1/M3 在目标大模型上的显存可行性与数值结果。

LoRA 当前通过的是不执行pickle的文件结构与契约验证；尚未在目标集群实例化
PEFT/vLLM完成真实加载，也未逐值扫描权重中的 NaN/Inf。entropy 消费端会从已记录的
action log scores 重算 entropy，但不会再次运行模型复算这些 logits。这些均属于真实
集成/复现实验阶段的边界，不应被描述为已经完成的模型级验证。

另有两个明确的实现边界：

1. `student_state_control.yaml` 与 `teacher_state_control.yaml` 是
   `confirmatory_use_allowed: false` 的设计约束模板；填入 `REQUIRED` 不会使它们
   自动变成可运行 control pipeline；
2. M3/SAGE 的 A1/A3 分析入口要求两个已完成的 (N=1) state-selection
   checkpoints/correction groups，但当前规范 pair/training launcher 只闭环 breadth `N=1`
   与 depth `N=3`。仓库尚未提供 A1/A3 的确认性上游生产链，M3 也未提供
   多 training-seed 最终汇总。

## 7. 最终判定

- **代码正确性**：在已执行的离线与单游戏集成验证范围内通过；关键输入身份与不可识别边界采用
  fail closed；
- **公式契合性**：状态/context、(q_T^V)、(B=MN)、(N)-dependent retention、
  分层方差、action CE、加权目标与机制估计量均有对应实现和验收门槛；
- **主链完整性**：fixed-budget breadth/depth 的采集→选择→annotation→数据→训练
  →评测→推断入口已连通；state-source control 与 A1/A3 生产链未完成；
- **论文结论有效性**：尚不能判定，必须按 `docs/REPRODUCTION.md` 在目标环境完成
  全链重跑；
- **旧结果处置**：只作探索性/历史证据，不得与 `omniopd-v1` 确认性结果混合。
