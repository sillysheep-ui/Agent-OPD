# 计划：以 veRL 原生 OPD 为主干，只保留一个必要钩子

目标：把训练侧从"我们手写的实现"切成"veRL 原生 OPD + 一个最小钩子"，让比较臂可被框架背书；
研究性的功能（状态选择、教师重采动作）之后再逐步加。本文只列实现清单与验收门槛。

## 一、原生 OPD 在 veRL v0.8.0 里的现状（已核对源码）

| 组件 | 位置 | 结论 |
|---|---|---|
| OPD 文档 | `docs/algo/opd.md` | 定义 GKD OPD（分布级 KL，直接反传）与 PG OPD（k1 作 reward + policy gradient） |
| 可跑示例 | `examples/on_policy_distillation_trainer/run_qwen3_*.sh` | `loss_mode=k1`、`use_policy_gradient=True`、`topk=32`、`use_task_rewards=False`、`lr=1e-5`、`epochs=15` |
| top-k 数据通道 | `verl/experimental/teacher_loop/teacher_manager.py:38` | `num_logprobs = topk if loss_settings.use_topk else 0`：**top-k 由配置触发，教师返回 `[S, K]`，无需改代码** |
| 蒸馏损失 | `verl/trainer/distillation/losses.py` | `forward_kl_topk` 直接读 `data["teacher_logprobs"]/["teacher_ids"]` |
| 教师打分调用点 | `verl/experimental/agent_loop/agent_loop.py:_compute_teacher_logprobs` | **教师拿到的是学生的 `prompt_ids + response_ids`** |

最后一行是本项目唯一必须偏离原生路径的地方：教师（Qwen3-14B，模板含 thinking 分支）与
学生（Qwen3-4B-Instruct-2507，模板无 thinking 分支）的渲染不同，实测在学生提示词下教师在
首个动作 token 的 argmax 是 `<think>`、给学生 token 的 logprob 是 −21.2；用教师自己的模板
渲染后变为 `Action` / −0.0。因此需要**一个**钩子：教师侧用自己的模板渲染提示词，再把分数
按学生布局重映射。

## 二、最小实现清单（做什么 / 不做什么）

保留（已是纯函数 + 已有测试）：

* `src/omniopd/opd_adapter.py` 的 `remap_teacher_scores_to_student_layout`、
  `align_teacher_rows_to_student_layout`（教师宽度与学生宽度对齐）。

删除/停用（这些是我们自造、且原生路径已有等价物）：

* 动作跨度 mask（`response_mask_for_action_span` 及调用处）—— 原生 OPD 用整段响应 `1/|y|`；
* `extract_strict_action_tokens` 的解析与记账（保留为诊断工具，不进训练主链）；
* 自写的 agent loop 除"教师提示词"以外的逻辑。

新增/改造（唯一必要的钩子）：

* 一个 `AgentLoopWorker` 子类，只覆盖 `_compute_teacher_logprobs`：
  1. 用**教师的 tokenizer + 模板**把同一段对话渲染成 `teacher_prompt_ids`
     （`enable_thinking=False` 会补出空 thinking 块）；
  2. 调用原生 `compute_teacher_logprobs_single(sequence_ids=teacher_prompt_ids + response_ids)`；
  3. 用上面的重映射把分数放到学生布局。

配置（全部为官方 flag，不再自造）：

```
distillation.enabled=True
distillation.distillation_loss.loss_mode=<见下>
distillation.distillation_loss.topk=64            # forward_kl_topk 时生效
distillation.distillation_loss.use_task_rewards=False
distillation.distillation_loss.use_policy_gradient=<见下>
```

## 三、按依赖顺序的两步走

**第 1 步（今天就能跑）：PG OPD，宽度 1，现有重映射直接可用。**

* `loss_mode=k1`、`use_policy_gradient=True`（官方 PG OPD：k1 作 reward + stop-gradient）。
* 教师提示词钩子按上面最小化。
* 数据：沿用 `opd_train_20260922_01/rows.jsonl`（学生响应在训练时现场采样，原生行为）。
* 验收门槛：① 教师首 token argmax 与 `Action` 一致；② 短跑后 `‖ΔW‖/‖W‖ ≥ 1e-3`
  （`scripts/diagnose_adapter_scale.py`）；③ 与冻结基线做同批配对评测。

**第 2 步（第 1 步通过后）：切成 GKD OPD（分布级，信息量更大）。**

* `loss_mode=forward_kl_topk`、`topk=64`（教师在原生通道自动返回 `[S, K]`）。
* 需要把重映射扩展为保留 K 列（当前断言只接受宽度 1），并补一条 K>1 的离线测试。
* 其余不变，仍与第 1 步同数据、同评测，便于比较两种 OPD 变体。

**第 3 步（研究性功能，慢慢改）：四条臂只允许"状态与动作选择"不同。**

* 在主干上加数据级选择（均匀 / 随机校正 / 所提选样），loss 与归一化保持常量；
* 所提臂需要新增"训练循环内教师重采动作 + 有效性筛选"（当前只有打分能力）。

## 四、风险与纪律

1. 冻结项（学生提示词、历史表示、评测协议、预算口径）在第 1 步之前不得再改，否则四臂不可比。
2. 每步只允许一个变量变化：第 1 步改主干与教师渲染，第 2 步只改 loss，第 3 步只改选择。
3. 每次启动前跑三条门槛（教师首 token、更新幅度、配对评测），并把结果写进台账。
