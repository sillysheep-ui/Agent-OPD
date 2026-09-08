# 旧代码逐文件审计索引

本表把两份 Word 中提取出的代码按“共同根因”分组。`legacy/` 中的文件保持原样，
用于追溯历史结果；“替代实现”才是新实验允许使用的入口。

## P0：旧结果必须从 state pool 起重跑

| 问题 | 受影响旧文件 | 实际后果 | 替代实现 |
|---|---|---|---|
| 初始 observation 被 task description 替代；step 后写回旧 observation；角色顺序错位 | `a1_collect.py`、`a3_collect.py`、`eval_zero_shot_v6.py`、`sft_teacher_collect.py`、`ctr_multiseed.py`、`collect_teacher_rollout.py` | 保存与查询的对象不是论文中的 \(z_t=(Task,H_t,O_t,\mathcal A_t)\) | `history.py`、`context.py`、`protocol.py` |
| 当前 observation 在截断后才追加 | 同上 | 当前 state 逃逸 token budget；可能出现连续 user 消息 | `TaskPreservingTruncator` 在完整 query 上截断，装不下当前 state 时直接失败 |
| 定义了 Teacher prompt，但请求继续使用 Student prompt | `a1_collect.py`、`a3_collect.py`、`a1_multiseed.py`、`ctr_multiseed.py`、`m1_probe_label.py`、`opd_collect.py` | 实际采样为 \(q_T(a\mid P_S,z)\)，不是声明的 \(q_T(a\mid P_T,z)\) | `query_teacher()` 只替换 system，并保持全部非 system 消息不变 |
| CTRL 同时改变 prompt、admissibles、thinking、修复规则和 learner context | `ctr_multiseed.py`、`ctrl_fix_invalid.py`、`ctrl_fix_invalid_v2.py`、`build_ctrl_data.py` | 不能解释成 Student-state 与 Teacher-state 的单变量对照 | `student_state_control.yaml`、`teacher_state_control.yaml`、`compare_control_protocols()` |
| Teacher 在错误 state 上生成动作，builder 又配到重新构造的另一 state | `sft_teacher_collect.py`、`build_sft_v6.py` | \(a_t^T\) 与训练 context 不对应；仅看 parquet 消息顺序无法修复 | 从统一 ledger 重新采集；`build_training_rows()` 不改变非 system state 内容 |
| next-token mask 使用 `[:, :-1]` | `fsdp_sft_trainer.py` | 监督前一个 prompt token并漏掉目标首 token；目标与公式不符 | `target_token_mask[:, 1:]`；`loss.py` 与 patched veRL trainer |
| backward 后才除 microbatch 数；每个 microbatch局部自归一 state weight | `fsdp_sft_trainer.py` | 梯度未被缩放；batch=1 时 \(1/K_s\) 完全抵消，随机batch权重和还产生自归一偏差 | uniform row sample 使用固定期望质量`rW/D`，每个microbatch只贡献可加 numerator |
| 分布式 sampler 丢样本或给重复 padding 正权重 | `fsdp_sft_trainer.py` | rank间步数/目标不一致；小数据集可能静默改变训练分布 | `ZeroPaddedDistributedSampler`：等长 shard，padding duplicate权重为0 |
| M1/M3 手工拼 prompt 与 action token | `m1_grad_alignment.py`、`m3_transferability.py`、`m3v2_transfer_retention.py`、`m3v3_transfer_retention.py` | 缺 assistant header、可能插入 BOS、EOT/thinking不同，测量的不是训练目标 | `encode_final_assistant_content()` 为训练、M1、M3唯一 token 契约，且排除模板 terminator |
| M3 A1/A3 聚合混组、交互项索引错误、CI上界取错 | `m3v3_stats.py` | group与交互系数无可解释性，所谓95% CI不是95% | `analysis/m3_stats.py` 使用具名列、correction-level回归、game cluster bootstrap与0.025/0.975分位数 |
| technical states在judge前删除；Teacher action dict被当字符串 | `sage_prep.py`、`sage_analyze.py` | Deterministic Strong不可达；disagreement几乎全错 | `analysis/sage.py` 统一 schema、保留 missing、technical state直接进入Strong |
| Position仅重跑Teacher分支，Student分支复用旧 `ep['won']` | `pos_counterfactual_v2.py` | 两分支除干预动作外还有代码、历史和运行版本差异 | `run_paired_counterfactual()` 独立replay两个对称分支并做state-fidelity gate |
| task type 从路径的 split/trial 层级解析 | `m2_representativeness.py`、`m3*_transfer*.py` 及其输入生成脚本 | task分层、matching与cluster解释错误 | 优先读取同 trial 的`traj_data.json.task_type`，仅以task-family目录作fail-closed fallback |
| checkpoint seed、解码seed和统计seed共用一个`seed` | `eval_zero_shot*.py`、`run_*eval*.sh`、`final_stats.py` | 多seed并非在同一评测随机机制下比较 | `training_seed`、`rollout_seed`、`bootstrap_rng_seed`分离且进入manifest |
| 只凭路径/文件名认定评测模型 | 所有旧评测启动脚本 | 可评错step、base或adapter，仍被汇总 | 评测绑定training manifest、最终`global_step`、base/adapter指纹、game-list manifest和vLLM版本 |
| M3直接将adapter目录交给`AutoModelForCausalLM`且未冻结完整score join | `m3v2_transfer_retention.py`、`m3v3_transfer_retention.py`、`run_m3v*.sh` | updated model可能加载失败或身份漂移；base/updated rows可能错配 | `score_action_logp.py`显式加载共同base+可选PEFT adapter，`assemble_m3.py`按不可变score ID一对一join |
| M1表与M3 adapter可脱离其生成链独立替换 | 旧M1/M3入口及初版重写入口 | 格式合法但来源错误的表/checkpoint仍可能产生貌似合理的机制结果 | M1要求完整未划分build audit；M3要求adapter是规范training manifest声明的final step且base一致 |
| Position从已截断query继续，Teacher-state还可能沿用Teacher system | `pos_counterfactual_v2.py`及初版重写逻辑 | continuation不是同一个冻结Student policy | 从`full_messages`恢复全部prefix并强制替换为(P_S)，每一步重新执行同一截断规则 |

## P1：实验或主张需要改名、降级或补实验

| 问题 | 受影响旧文件 | 处置 |
|---|---|---|
| 原A2保持M不变而把N从1增至3，Teacher预算和optimizer steps同时增加 | `a1_multiseed.py`、`build_a2_data.py`、`run_a2_train.sh`、`run_n3_v2_train.sh` | 旧结果只能称“额外预算depth”；新协议固定为150×1对50×3，且显式固定训练步数 |
| v2 builder输出与launcher输入断开，只重训N3 | `build_a2_data.py`、`run_a2_train.sh`、`run_n3_v2_train.sh` | 旧N1/N3不可直接比较；新launcher只接受显式数据路径与总步数 |
| valid filtering改变Teacher分布 | `build_a1_data*.py`、`build_a2_data.py`、`build_a3_clean.py`、`build_a3_data.py`、`build_a4_data.py`、`build_a5_data.py`、`build_ctrl_data.py`、`build_sft_v6.py` | 新代码保留全部attempt；训练目标明确命名为acceptance-conditioned game balance，并报告每game与每sample接受率 |
| A1与A3重新rollout，未共享同一冻结state pool | `a1_collect.py`、`a3_collect.py` | selection差异混入rollout/version差异；确认性比较必须从同一 immutable ledger派生 |
| A3 append输出非幂等；合并不检查重复/缺shard | `a3_collect.py`、`merge_a3.py`、`merge_a1s.py` | 新入口拒绝覆盖，`merge.py`稳定排序并对重复record立即失败 |
| prior-round exclusion只读顶层字段 | `a1_collect.py` | canonical嵌套state中的game可能未排除 | `record_game_id()`兼容顶层/嵌套与旧`gamefile` |
| Round0 collector与builder字段不一致；full target包含Teacher文本 | `opd_collect.py`、`opd_prepare.py`、`run_round0.sh` | pipeline可能KeyError，且违反action-only假设；新schema只允许最终assistant action目标 |
| M1 reference由selection补集定义且不是最终成功率梯度 | `m1_probe_label.py`、`m1_grad_alignment.py` | 只能称surrogate alignment；新代码要求预注册、game-disjoint probe，主报dot/norm/cosine |
| M2使用turn-pooled baseline、只看边际、未纳入valid filtering | `m2_representativeness.py` | 不能代表论文的 \(q_{GB}\) 或 \(q_{train}\) | `representativeness.py` 先按game求分布再平均，并做配对game-cluster bootstrap；仍明确为诊断而非因果utility |
| M3使用整套checkpoint的概率变化归因单条correction；旧BOW cosine错误 | `m3*_transfer*.py` | 只能称action-imitation transfer；需要crossed checkpoint评估或one-correction update | `analysis/transfer.py` 修正cosine/neighbor gate，`analysis/m3_stats.py`只做限定后的统计 |
| SAGE定向分层样本直接报告总体比例，invalid D进入分母 | `sage_audit_prep_v2.py`、`audit_analysis.py`、`sage_analyze.py` | 未加权结果不能推广总体 | `design_weighted_game_balanced_indicator()`要求已知inclusion probability并使用Horvitz–Thompson估计 |
| 旧bootstrap只重采game，不传播训练seed；seed42来源不同 | `a1_vs_ctrl_fixed.py`、`boot_verify.py`、`mc_final.py`、`final_stats.py` | CI只能条件于固定checkpoint | `paired_hierarchical_bootstrap()`对配对的seed×game交叉因素重采样；历史异源seed仍不得混用 |
| matched-count只匹配parquet行数 | `build_mc_data.py`、`mc_final.py` | 不能排除API预算、state/token/step与任务构成混杂 | `exact_stratified_sample()`只支持预注册strata且support不足时失败；训练步数另行固定 |
| LoRA合并精度和base/tokenizer未统一验证 | `merge_lora.py`、`merge_a1s.py` | checkpoint差异可能混入数值误差 | 统一FP32 merge、记录base/tokenizer hash，并在固定prompt上做adapter/merged logits等价测试后才发布 |
| 文件系统返回顺序直接进入shuffle/抽样 | 多个collector与game-list脚本 | 同seed在不同机器仍会选择不同games | `stable_shuffled()`先排序再以独立seed洗牌 |
| entropy score与selection只按`state_hash`松散连接 | `score_a1_entropy.py`、A3/A4/A5 builders | 可混入另一pool、模型或token定义的score | score/selection双manifest绑定pool、代码、生成该pool的同一Student/Tokenizer及action-content token契约；从逐action log score重算entropy |
| SAGE judge失败无一致missing语义、population来自未知pool | `sage_judge.py`、`sage_analyze.py` | 失败可能被删、误填或与错误总体做HT估计 | label manifest绑定correction/pool；失败保留`None`，总体估计在缺失或top-k概率未知时明确不可识别 |
| 未固定veRL来源且可误导入旧patched trainer | `run_verl_sft*.sh` | 不同节点运行不同loss/precision代码 | launcher直接执行仓库trainer，要求veRL 0.4.1声明与Git revision，并记录实现文件hash |

## 仅作为运行编排、但受上游错误污染的文件

以下脚本的主要问题不是自身算法，而是调用了上述旧数据、旧trainer或硬编码路径，因此也不能
作为新实验入口：

- 训练：`run_a1_train.sh`、`run_a3_train.sh`、`run_a4_train.sh`、
  `run_a5_train.sh`、`run_ctrl_train.sh`、`run_mc_train.sh`、`run_seed_train.sh`、
  `run_verl_sft*.sh`；
- 评测：`run_a1_eval.sh`、`run_a2_serial_eval.sh`、`run_a3_eval.sh`、
  `run_a4_eval.sh`、`run_a5_eval.sh`、`run_ctrl_eval.sh`、`run_mc_eval2.sh`、
  `run_n3v2_eval.sh`、`run_seed_eval.sh`、补充文档中的`run_*_eval*.sh`；
- 服务启动：`start_vllm_*.sh`、`start_3vllm_seeds.sh`、`start_4vllm_134.sh`、
  `start_4vllm_d3.sh`；
- 旧汇总/检查：`score_a1_entropy.py`、`pos_data_check.py`、`pos_gate_v2.py`、
  `verify_lossmask.py`、`verify_thinking.py`、`transcribe_audit.py`。

这些文件全部保存在`legacy/`，但README明确禁止把它们当作canonical runtime。

## 对规范重写版的第二轮交叉审计

第一轮重写消除了旧 Word 代码中的主要算法错误，但“函数测试通过”仍不足以证明跨阶段
实验身份闭环。第二轮以可替换输入为对手模型，继续发现并关闭以下缺口：

| 缺口 | 涉及规范文件 | 修复后的拒绝条件 |
|---|---|---|
| annotation manifest 可自报错误的 (M,N,B)、valid/invalid totals 或 selected states | `validation.py`、`build_data.py`、`analyze_m2_representativeness.py`、`run_position_counterfactual.py` | 消费者从 corrections 实体重算并要求逐字段相等 |
| 两臂只比较配置，未绑定 realized annotation bytes 和效应方向 | `validate_experiment_pair.py`、`evaluation.py`、`run_verl_train.sh` | 仅接受恰好 breadth/depth 两臂的 schema-v2 pair；绑定完整成员清单、共同契约和 `breadth_minus_depth` |
| state pool 的 behavior Student 可与训练 base 不同 | `collect_state_pool.py`、`evaluation.py`、`run_verl_train.sh` | pair/launch 继承 pool behavior artifact 与Tokenizer内容身份；不相等即拒绝 |
| launch 目录存在就被当作完成训练 | `run_verl_train.sh`、`evaluation.py` | 仅在 trainer 与日志管道成功后生成 completion manifest，并绑定最终step、checkpoint、resolved config与日志 |
| vLLM alias 与实际服务模型可脱离 | `run_vllm_state_pool.sh`、`run_vllm_eval.sh`、`run_vllm_position.sh`、`service_attestation.py` | 现场核验wrapper子进程、PID、argv、端口、runtime及模型/Tokenizer/checkpoint内容 |
| M1只用有效action rows检查game-disjoint；零梯度cosine被写成0 | `analyze_m1_gradients.py`、`analysis/gradient.py` | 用完整selected-game集合检查；零范数时明确不可识别/拒绝 |
| M2 turn-pooled/单game bootstrap或清单实体不一致 | `representativeness.py`、`analyze_m2_representativeness.py` | 先game内归一再game等权；共同支持至少两个games；重算correction契约 |
| M3六格可共同使用与训练不同的Tokenizer | `score_action_logp.py`、`assemble_m3.py` | updated cells与各自训练Tokenizer一致，base cells与共同训练Tokenizer一致 |
| SAGE缺少A3−A1检验且 (U_g) 混入Student-invalid分母 | `analysis/sage.py`、`analyze_sage.py` | 主报`student_valid=1`条件指标；共同game support上做配对差与cluster bootstrap |
| SAGE correction manifest可自报伪造的state/count/validity | `judge_sage.py`、`validation.py` | 盲评前从correction records重算M/N/B、selected hashes、逐game计数、valid/invalid与selection policy |
| Position或评测汇总可接受与records/traces矛盾的派生字段 | `run_position_counterfactual.py`、`aggregate_evaluations.py` | 从源records/traces重算 inclusion/计数/逐game success，拒绝缓存字段漂移 |
| 只冻结game路径，不冻结真实环境 | `freeze_game_list.py`、`environment_provenance.py`、各rollout入口 | 绑定trial内容、runtime版本、显式TextWorld per-game seed；无法设seed即拒绝确认性运行 |
| selection/correction 生成器遗漏下游必需的 behavior Student 契约 | `select_states.py`、`annotate_states.py`、`validation.py` | 两级manifest均从已验证state-pool统一派生`behavior_student`；pool→selection→annotation→pair端到端测试必须通过 |
| selection manifest 可自报逐game数量、状态集合或错误的 (m/T_g) | `select_states.py`、`annotate_states.py`、`validation.py` | 花费Teacher预算前从冻结pool复建逐game精确 seeded-uniform/top-score 子集；uniform复算 (m/T_g)，top-score不得携带design probability |
| final checkpoint 仅检查目录非空，未证明是可加载LoRA | `fsdp_sft_trainer.py`、`run_verl_train.sh`、`evaluation.py` | trainer与completion schema v2验证PEFT语义、唯一safetensors、Tokenizer、launch rank/alpha、成对A/B tensor和target覆盖；消费者重解析当前目录 |
| state-pool的运行中服务证明未进入behavior Student契约 | `collect_state_pool.py`、`validation.py`、`evaluation.py` | 规范化service attestation及其摘要沿selection/correction/pair传播，训练与Position复核同一模型、Tokenizer、runtime和context容量 |
| top-score只绑定缓存score，未证明其动作级定义 | `score_entropy.py`、`select_states.py`、`annotate_states.py`、`validation.py` | score rows与pool一一对应，game/turn/admissibles一致，模型/Tokenizer一致，并从完整有限action log scores重算entropy |

这一区分很重要：`legacy/` 记录的是原始实现问题；上表记录的是规范实现自身在独立复核中
被发现并修复的“验证器缺口”。二者都属于最终审计的一部分。
