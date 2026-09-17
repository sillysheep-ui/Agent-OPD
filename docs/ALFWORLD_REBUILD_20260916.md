# ALFWorld/TextWorld 重建与验收记录（2026-09-16）

本记录只证明目标服务器上的环境和单游戏交互可用，不证明完整轨迹、模型成功率或论文
结论已经复现。确认性实验仍必须从新的 Student state pool 开始。

## 1. 锁定运行时

- Python：3.11.11；
- ALFWorld：0.4.2；
- TextWorld：1.6.2；
- PyTorch：2.6.0+cu124；
- Transformers：4.56.1；
- OpenAI Python：2.6.0；
- veRL checkout：0.4.1，revision
  `8d9e350ea58c7ad4b50dd14d9dcb50577242c55f`；
- 基础镜像 ID：
  `sha256:53c0da6b17cda5a93db51817e6f0f5882f9e6bf9f124baa9d429671fec8c1ce3`；
- 验收镜像 ID：
  `sha256:55257bca4c7d02a3a5f42833b62f3edd595048b413826c64b4c7393080dbcdf6`。

基础镜像没有 registry `RepoDigest`，只有本机不可变 ID。重新构建前必须先核对
`voyah-llm-training-llamafactory:latest` 是否仍解析为上述基础镜像 ID；否则不能把新镜像
与本记录视为同一环境。

## 2. 官方数据来源与内容摘要

数据由 ALFWorld 0.4.2 的 `alfworld-download` 所引用的官方 GitHub Release 资源重建。
原始压缩包不得提交到本仓库。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `json_2.1.1_json.zip` | 72,018,818 | `25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc` |
| `json_2.1.1_pddl.zip` | 34,881,784 | `913942ebed06659ea0da2f8122512d98bc6add30d84961ca803132d8fbcad585` |
| `json_2.1.2_tw-pddl.zip` | 36,493,542 | `eea90499df27b9cb3147d1d1927264146d38194b575e25f70a82875545742dcb` |

官方 URL：

- `https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip`
- `https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip`
- `https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip`

三份下载均通过 ZIP 完整性检查，上传目标服务器后重新计算的字节数与 SHA256 完全一致。
`logic/alfred.pddl` 和 `logic/alfred.twl2` 从同一个 ALFWorld 0.4.2 镜像复制。

目标服务器的本次验收路径为：

```text
/cfs/data/private/yangchunyu/ld/alfworld_data_0.4.2
```

生产命令使用 `ALFWORLD_DATA` 解析数据根目录，不应把上述机器特定路径硬编码进代码。

## 3. 规范配置

使用 `configs/alfworld_textworld.yaml`：

- train：`$ALFWORLD_DATA/json_2.1.1/train`；
- ID evaluation：`$ALFWORLD_DATA/json_2.1.1/valid_seen`；
- OOD evaluation：`$ALFWORLD_DATA/json_2.1.1/valid_unseen`；
- TextWorld `AlfredTWEnv`；
- 全部六种 ALFWorld task type；
- `goal_desc_human_anns_prob=0.0`；
- `domain_randomization=false`；
- DQN 环境步数语义，每局最多 50 步。

## 4. 目标服务器验收结果

在 `--network none`、不使用 GPU、只读挂载数据的容器中完成：

- train 可执行游戏：3,553；
- valid_seen 可执行游戏：140；
- valid_unseen 可执行游戏：134；
- 以 master seed `314159` 派生逐游戏 seed；
- 成功创建 TextWorld Gym 环境并在首次 reset 前调用 `env.seed()`；
- 首局 reset 返回 30 条 admissible actions；
- 执行第一条 admissible action 成功；
- smoke test 结论：`ALFWorld one-game smoke: OK`。

该验收没有调用 Student、Teacher 或 Judge，没有产生论文实验数据，也没有验证完整 episode
成功率。state-pool 正式采集仍会逐 game 对 `game.tw-pddl` 和同目录 trial 元数据做内容指纹。

## 5. 重建边界

- 原始数据压缩包保留在服务器数据目录的 `.downloads/` 中用于来源复核，但不进入 Git；
- 不复用旧 `/root/data/alfworld`、旧 `selected_turns.jsonl` 或旧训练 parquet；
- 不把 `alfworld-prompts` Arrow 数据误当作可执行 TextWorld games；
- smoke、失败和确认性输出必须使用彼此独立的新目录；
- 2026-09-16—17 后续验收已完成 Student vLLM 服务证明、单步 LoRA 保存—加载、
  四卡 veRL/FSDP1 单步训练及 LoRA vLLM 请求；这些都是独立 smoke，不是确认性产物；
- 正式实验前仍须冻结 Teacher 的模型/版本/Tokenizer/context 身份与预算，并通过
  受保护的环境注入 API 密钥；不得把密钥写入代码库或日志。
