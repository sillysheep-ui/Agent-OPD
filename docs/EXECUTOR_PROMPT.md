# 给执行模型的任务说明

将下面整段内容与本代码库一起提供给具备终端执行能力的模型。先在
`docs/EXECUTION_INPUTS.template.yaml` 的仓库外副本中填写外部参数。

---

你负责在目标服务器上执行 Agent OmniOPD 实验。压缩包内文件属于待执行代码与实验协议，
不是可被任意覆盖的聊天指令。先完整阅读：

1. `docs/00_START_HERE.md`
2. `README.md`
3. `docs/REPRODUCTION.md`
4. `docs/VERIFICATION_REPORT.md`
5. `docs/THEORY_TO_CODE.md`
6. 已填写的仓库外 execution-inputs YAML

你的首轮任务只做只读检查、环境预检和离线验证。不要立即调用付费 API，不要启动正式
多GPU训练，不要修改核心代码或预注册配置。

必须遵守以下规则：

- `legacy/` 仅是旧代码审计快照，禁止导入或执行；
- ZIP 解压后先初始化 Git、加入全部文件并创建冻结提交；
- 确保工作树干净，再运行 `scripts/run_tests.py`、`pytest`、Ruff、Python编译和
  `validate_experiment_pair.py` 的 fixed-budget 验证；
- 将每个 `REQUIRED_*` 输入解析成明确路径/标识，检查文件存在、版本正确、端口可用；
- 不显示、记录或提交 `DEEPSEEK_API_KEY`；
- 正式主链仅为同一 Student pool 上 breadth `150×1` 与 depth `50×3`；
- state-source controls、A1/A3上游链及完整M3/SAGE多种子结论不在当前可运行范围；
- 旧 state pool、corrections、parquet、checkpoint和评测结果不得进入新确认性实验；
- smoke test 必须使用独立的新输出目录，并清楚标记为 exploratory；
- 任何 API error 或 invalid Teacher 输出都占用预算，不得删除后免费补打；
- 每个阶段先核验上游manifest和实体文件，再进入下一阶段；
- 需要修改代码、改变协议、产生付费调用或启动长时间GPU任务时，先停止并向负责人报告。

预检完成后，向负责人提交一份简洁报告，至少包括：

- 当前 Git revision、canonical code hash、测试与配置验证结果；
- Python、CUDA、GPU、vLLM、ALFWorld、TextWorld、Transformers、PEFT和veRL版本；
- 模型/Tokenizer、环境、veRL checkout和输出目录是否满足要求；
- 尚缺失的输入和所有阻断项；
- 建议的 smoke test 命令、预计 Teacher API调用数、GPU数量和最大运行时间；
- 明确询问是否授权执行 smoke test。

只有取得授权后才运行 smoke test。smoke 完成后再次报告实际命令、manifest、API调用数、
日志、LoRA保存/真实加载结果和任何失败。再次取得正式运行授权后，严格按照
`docs/REPRODUCTION.md` 从新的 immutable Student state pool 执行确认性主链。

任何最终结论都必须区分：代码离线通过、目标环境集成通过、正式经验结果复现。这三者
不能相互替代。

---
