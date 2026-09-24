# veRL 原生 OPD 的结果（2026-09-24）

## 结论

在修正了教师渲染与目标函数、并按 LoRA 重新标定学习率之后，**传统 OP（OPD）这一臂首次显著超过
冻结基线**：同批 140 局配对，**66/140 = 47.14% vs 50/140 = 35.71%**，净 **+16 局**，
McNemar 精确检验 **p = 0.0328**。权重只动了 **0.12%**（见下），因此这不是"把模型练坏了"的假象，
而是监督目标变对之后的小幅有效更新。

## 运行身份

| 项 | 值 |
|---|---|
| 运行 | `omniopd_runs/opd_native_20260924_01` |
| 实现 | `src/omniopd/opd_native.py` + `opd_native_layout.py`（veRL 原生 OPD + 两个小覆盖） |
| 卡 | 4 卡：学生 GPU 0-1，教师 1 副本 × TP2 在 GPU 2-3 |
| loss | 官方 PG OPD：`loss_mode=k1`、`use_policy_gradient=True`、`use_task_rewards=False`、官方 clamp（±10） |
| 超参 | LoRA r16/α32、**LR 3e-5**、86 步（1 epoch）、batch 64、rollout 温度 1.0 |
| 落盘 | `global_step_43`、`global_step_86`、`completion_manifest.json` |
| 权重变化 | `‖ΔW‖/‖W‖ = 1.23e-3`（`diagnose_adapter_scale.py`，252 个模块） |
| 评测 | 同批同服务：LoRA 在线加载 `opd` 与 `base4b`，140 局 valid_seen、30 轮、贪婪、AgentBoard 提示词 |

## 评测结果

| 模型 | valid_seen | 无效回合 | 总步数 |
|---|---|---|---|
| **OPD（本次）** | **66/140 = 47.14%** | 686 | 2944 |
| 冻结 base（同批） | 50/140 = 35.71% | 420 | 3208 |

逐局配对：双赢 33、**仅 OPD 赢 33**、仅 base 赢 17、双输 57 → 净 +16，p = 0.0328。

| 任务族 | OPD | base |
|---|---|---|
| look_at_obj_in_light | 8/13 | 8/13 |
| pick_and_place_simple | 20/35 | 19/35 |
| **pick_clean_then_place_in_recep** | **12/27** | **2/27** |
| pick_cool_then_place_in_recep | 13/25 | 10/25 |
| pick_heat_then_place_in_recep | 5/16 | 5/16 |
| pick_two_obj_and_place | 8/24 | 6/24 |

提升集中在过去所有基线都最弱的 `pick_clean_then_place_in_recep`（2→12），其余任务族小幅或持平。
行为侧：OPD 的无效动作回合更多（686 vs 420）但赢更多，形态上是"更敢尝试"。

## 与之前“无效果”结果的对照

| 版本 | 教师渲染 | 目标函数 | LR / 权重变化 | 结果 |
|---|---|---|---|---|
| `opd_standard_20260923_18`（旧） | 学生模板（错，首 token 目标=`<think>`，−21 nats） | `k2` 单采样 | 1e-6 / 7.8e-5 | 49/140，配对 −2 局，p=0.75 |
| **`opd_native_20260924_01`（本次）** | **教师自身模板（+4 token 的 thinking 块）** | **官方 PG OPD（k1 作 reward）** | **3e-5 / 1.23e-3** | **66/140，配对 +16 局，p=0.033** |

三处改动缺一不可：教师目标不再被格式偏差主导；目标函数换成官方变体；学习率按 LoRA 重新标定到
`‖ΔW‖/‖W‖` 约 1e-3 量级。**"OPD 在这个规模上没效果"的旧结论因此作废**，它是实现缺陷的产物。

## 必须随结果报告的限制

1. 单批次、单种子环境（140 局）；这一批的 base 为 50/140，而历史同协议测量在 45–51 之间波动
   （±4 点）。本次效应 +11.4 点大于该漂移，且配对检验显著，但仍建议后续扩到
   valid_seen + valid_unseen 或多种子复核。
2. 这是**传统 OPD 比较臂**（学生自身 token 监督），不是所提的"教师重采动作 + 选样"方法。
3. 训练仍是固定单步状态池（1 epoch），不是多轮轨迹在线训练。
4. LR/步数是按 LoRA 标定的，与论文（若为全参数微调）不可直接换算。
