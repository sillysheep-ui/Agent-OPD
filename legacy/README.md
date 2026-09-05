# Legacy audit snapshots

这里保存从用户提供的两份 Word 文档机械提取的历史代码，目的是保证问题定位和
provenance 追溯。文件未经修复，包含已确认的 P0 问题。

不得：

- 将本目录加入 `PYTHONPATH`；
- 从本目录启动训练或评测；
- 用这里的脚本生成新的确认性实验结果；
- 把旧结果自动标记为由 `omniopd-v1` 生成。

规范实现位于 `src/omniopd` 与顶层 `scripts`。

主文档明确标注 `build_sft_v6.py` 缺失；`main_document` 中保留了对应的
`.MISSING.txt` 占位说明，实际补充版本位于 `supplement_document/build_sft_v6.py`。
