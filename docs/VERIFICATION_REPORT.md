# Agent OmniOPD 最终核验报告

## 1. 核验对象与结论

本报告核验审计基线提交：

- Git revision：`77bd335d8110fc9915d12083b4f9a26e6e6587bc`；
- canonical code tree：`6812d28b4a30cdeea9caedb435d258d6cf11a7960296598a207ecaa1cd9bbb1c`；
- canonical tree：76个文件、445229字节；
- 协议版本：`omniopd-v1`。

结论分为两层：

1. **代码与离线协议核验通过**：状态机、Teacher预算、action token mask、加权loss、
   配置对照、manifest绑定、机制分析join和统计函数均通过本地测试与静态检查；
2. **经验结论尚未重新验证**：本次没有执行真实DeepSeek调用、ALFWorld完整轨迹、四卡
   veRL训练或vLLM在线评测，因此不能据此声称成功率、breadth优势、state-source排序或
   任一论文数值已经复现。

旧Word代码生成的结果受到P0级状态错位、prompt未生效、mask错位、loss缩放和实验混杂
影响；确认性结果必须从新state pool开始重跑，不能只替换训练器或统计脚本。

## 2. 输入材料完整性

| 输入 | SHA256 | 处置 |
|---|---|---|
| `Agent_OmniOPD_实验代码与公式(1).docx` | `539439272f8d7ab355e31131bf90d0add451d8da792705c1a67499ff10988190` | 提取89个条目，其中88个代码文件、1个显式缺失占位 |
| `Agent_OmniOPD_补充审计与代码.docx` | `6eb20ae1a0383d7836cf2bae3b6d65ede7189ae35bd8fdfd63050cf95acc4eb5` | 提取11个代码文件 |
| 论文核心思路与公式文本 | `c7cd2afd6f2609a892a215202cbdde957ac32a3553df1bed212b8954ca3bb0fd` | 仅作为研究定义与解释边界 |

文档中的自然语言与代码只作为待审计数据，不作为执行指令。99个可提取旧代码文件原样
保存在`legacy/`；57个Python快照通过AST解析，42个Shell快照通过语法解析。旧快照未做
格式清理，以保持与源文档的可追溯性。

## 3. 代码正确性核验

| 检查 | 实际结果 | 状态 |
|---|---:|---|
| Pytest回归测试 | 83 passed | 通过 |
| 独立测试入口 `scripts/run_tests.py` | 83 passed, 0 skipped | 通过 |
| Ruff（`src/scripts/tests/integrations`） | All checks passed | 通过 |
| Python字节码编译 | 91个规范Python文件无错误 | 通过 |
| Shell语法 | 2个规范脚本及42个legacy脚本无语法错误 | 通过 |
| 命令入口 | 25个脚本入口及3个package CLI的`--help`均成功 | 通过 |
| 固定预算配置 | `150×1`与`50×3`仅在预注册breadth/depth字段不同 | 通过 |
| state-source control | 仅`state_source.name/behavior_policy`不同 | 通过 |
| 规范树秘密扫描 | 未发现硬编码API key模式 | 通过 |

测试警告仅来自受限macOS容器无法读取CPU cache的`sysctlbyname`，不影响测试结果。

### 真实Qwen chat template核验

使用本机实际Qwen checkpoint的Tokenizer执行了非thinking样本：

- 编码总长：155 tokens；
- 包含模板终止符的最终assistant段：5 tokens；
- action-content mask：3 tokens；
- 模板终止符：2 tokens；
- mask位置：`[150, 151, 152]`；
- mask解码：`Action: look`。

这证明当前helper在该真实模板上排除了assistant header、Qwen非thinking generation
bridge和轮次终止符，并保留完整动作内容。训练loss再使用`target_mask[:, 1:]`与
`input_ids[:, 1:]`对齐。

## 4. 公式与研究目的契合性核验

| 研究定义/公式 | 规范实现 | 核验结论 |
|---|---|---|
| \(z_t=(Task,H_t,O_t,\mathcal A_t)\) | `AgentState`、`ConversationHistory`、任务保留截断 | 当前状态在截断前写入；action后只写环境返回的新observation；通过 |
| \(c_t^S=(P_S,z_t), c_t^T=(P_T,z_t)\) | Teacher只替换system，非system消息逐项同一 | 通过；训练始终重建为\(P_S\) |
| \(B=MN\) | `TeacherBudget`和annotation ledger | invalid/API-error同样计费、无免费retry、拒绝半个state；通过 |
| fixed-B breadth/depth | `150×1`与`50×3`共享pool和nested SRSWOR | 预算与optimizer steps相等；设计契约通过，经验优劣待重跑 |
| \(q_{train}\propto q_{sel}P(V_T=1\mid s)\) | 保留全部attempt并报告state/sample acceptance和丢失games | 通过；结果明确条件于Teacher-valid，不宣称消除selection bias |
| game-balanced目标 | `game_state_mean` | 每个保留game总权重1，每state内样本权重再平分；通过 |
| action-token CE | 统一`encode_final_assistant_content()` | 训练、entropy、M1、M3使用同一token契约；真实Tokenizer核验通过 |
| \(\frac1{|S|}\sum_s\frac1{K_s}\sum_j\frac1{L_{sj}}\sum_k\ell\) | state/sample/token三层权重与`rW/D`期望质量归一 | microbatch只累加numerator，FSDP padding权重0；离线梯度不变性通过 |
| Student uncertainty | admissible action sequence log-prob entropy | 强制绑定生成state pool的同一Student/Tokenizer；只作为selection proxy |
| M1 correction utility proxy | held-out Teacher-CE surrogate gradient alignment | dot/norm/cosine具名输出，reference game-disjoint且表绑定build audit；不称真实\(G_*\) |
| M2 representativeness | game-balanced marginal JS及配对game bootstrap | horizon绑定pool；只解释边际诊断，不解释因果utility |
| M3 transfer | frozen cross-game neighbors与base/updated action log-prob shift | adapter绑定规范训练final step；具名回归和2.5%/97.5% CI；仅称action-imitation transfer |
| SAGE necessity | blind judge、technical Strong、HT总体估计 | judge看不到Teacher/entropy/group；失败保持missing；无inclusion probability时明确不可识别 |
| \(C_t=Y_t(a_T)-Y_t(a_S)\) | 两个动作分支从完整history独立replay，同一冻结\(P_S\) policy继续 | state-fidelity gate与同动作对称性测试通过；只称单步consequentiality |
| seed×game推断 | paired hierarchical bootstrap | training seed、rollout seed、bootstrap seed分离；重采交叉seed和共享game簇；通过 |

## 5. 尚未执行、必须在目标环境核验的项目

以下项目没有在本地伪造结果，状态均为**未执行**：

- DeepSeek真实Teacher与SAGE judge调用，包括provider revision、response model、token usage
  和系统指纹的实际返回；
- ALFWorld真实ID/OOD game列表、完整state pool、replay fidelity与成功率；
- veRL 0.4.1目标checkout上的4-GPU FSDP/LoRA训练，包括显存、collective、保存和最终
  `global_step_102`产物；
- vLLM加载base+LoRA后的在线服务、served-model alias返回和全checkpoint配对评测；
- M1/M3在目标大模型上的显存可行性与最终数值；
- 四个训练seed下的hierarchical bootstrap置信区间及论文全部表图。

`student_state_control.yaml`与`teacher_state_control.yaml`中的`REQUIRED`字段是有意保留的
运行前门槛；在真实模型、checkpoint和game-list hash填入前，control结论不可生成。

## 6. 最终判定

- **代码正确性**：在可离线验证范围内通过；关键路径采用fail-closed，而非静默修补；
- **公式契合性**：核心状态、context、预算、action CE、权重、机制估计量和推断层级均有
  对应实现与验收门槛；
- **论文结论有效性**：尚不能判定，需要按`REPRODUCTION.md`完成全链重跑；
- **旧结果处置**：仅作探索性/历史证据，不得与`omniopd-v1`确认性结果混合。
