# 传统逐 Token OPD 首个完整结果（2026-09-23）

## 结论一句话

在冻结协议下（86 个优化步、LoRA rank 16、LR 1e-6、1 epoch），传统逐 Token OPD
**没有超过冻结基线**：49/140 vs 51/140，逐局配对净差 −2 局，McNemar 精确检验
p = 0.754。同一批评测里，冻结基线的复测值比早先的 45/140 高了 6 局，说明这个
评测规模下的复测波动与我们想检测的效应同量级。

## 运行身份

| 项目 | 值 |
|---|---|
| 训练运行 | `omniopd_runs/opd_standard_20260923_18`（第 18 次，八卡：学生 0-3、教师 4-7） |
| 优化步数 | 86（1 epoch，5524 条固定状态行，batch 64） |
| Student | Qwen3-4B-Instruct-2507（LoRA r=16, α=32），LR 1e-6，K2 蒸馏，温度 1.0 |
| Teacher | 冻结 Qwen3-14B，teacher-scores-student-prefix（标准 token OPD） |
| checkpoint | `checkpoints/global_step_86`（24 文件，18.06 GB，tree sha256 见完成清单） |
| 中途点 | `checkpoints/global_step_43`（落盘计划修复后新增） |
| 完成清单 | `completion_manifest.json`（含 checkpoint 与 train.log 的哈希） |
| 转换产物 | `omniopd_runs/opd_adapter_opd_standard_20260923_18/`（504 张量、252 模块，与基座逐层对账通过） |
| 服务方式 | vLLM LoRA（同一服务同时提供 `opd` 与 `base4b`），140 局、30 轮、贪婪、AgentBoard 提示词 |

## 结果

| 模型 | valid_seen | 与本次配对的净差 |
|---|---|---|
| OPD Student（86 步） | **49/140 = 35.00%** | −2 局 |
| 冻结 base Student（同批同服务复测） | **51/140 = 36.43%** | 基线 |
| 冻结 base（2026-09-21 阶梯评测的旧值） | 45/140 = 32.14% | 复测漂移 +6 局 |

逐局配对（140 局、game_id 与环境种子逐一致，已校验）：

| 类别 | 局数 |
|---|---|
| 两者都赢 | 45 |
| 仅 OPD 赢 | 4 |
| 仅 base 赢 | 6 |
| 两者都输 | 85 |

McNemar 精确检验（不一致 10 局，净 −2）：p = 0.754。

| 任务族 | OPD | base | 仅 OPD 赢 | 仅 base 赢 |
|---|---|---|---|---|
| look_at_obj_in_light | 6/13 | 8/13 | 0 | 2 |
| pick_and_place_simple | 19/35 | 18/35 | 2 | 1 |
| pick_clean_then_place_in_recep | 4/27 | 5/27 | 1 | 2 |
| pick_cool_then_place_in_recep | 11/25 | 10/25 | 1 | 0 |
| pick_heat_then_place_in_recep | 4/16 | 4/16 | 0 | 0 |
| pick_two_obj_and_place | 5/24 | 6/24 | 0 | 1 |

## 训练强度诊断

对 `opd_standard18.log` 的 86 个 step 统计：

| 指标 | min | median | max |
|---|---|---|---|
| actor/grad_norm | 7.51 | 32.2 | 128 |
| actor/distillation/loss | 2.31 | 6.33 | 13.7 |
| actor/entropy | 0.0167 | 0.0938 | 1.36 |
| response_length/mean | 7.59 | 26.4 | 177 |

在一条**真实 OPD 状态提示词**上，用 float32 比较基座 / 挂 adapter / 合并后再加载三组
logits：adapter 对最后一层 logits 的最大影响是 **0.326**，而 bf16 导出的合并误差是
**0.490**。也就是说，这个预算训出来的 adapter 对模型行为的改变量，还小于导出精度本身
带来的差别；评测上看到"没有变化"与此一致。

## 必须随结果一起报告的限制

1. **评测规模不足**：140 局下，同配置复测就漂移了 6 局（45 → 51）。任何 ±3 点量级的
   结论都不可采信。
2. **协议偏差**：标准 OPD 臂在无 `Action:` 行的回合按整条输出监督（约 0.2% 频率，
   见 E14）。严格分支仍 fail-closed。
3. **服务路径**：本次用 vLLM LoRA 在线加载 adapter（与基座同服务、同批次）。合并权重
   兜底路径虽已验证，但因其 bf16 导出误差大于本次 adapter 影响，**不适合**用来说明
   本结果。
4. **仍未验证**：valid_unseen 未评测；未做多环境种子重复。

## 下一步（需要决策，不在本记录内擅自执行）

1. 提高统计功效：评测扩到 valid_seen + valid_unseen（274 局）或每局多种子。
2. 提高训练信号：增加 epoch（86 → 258 步）或提高 LR（1e-6 → 1e-5）。这会偏离当前冻结
   协议，属于实验设计决定，需要明确后再改。
3. 先做便宜的"训练有效性"检查：在一批固定状态上比较 base 与 opd 的贪婪动作改变了多少，
   比直接跑 140 局快得多，可在扩展预算前确认 OPD 是否真在改变行为。
