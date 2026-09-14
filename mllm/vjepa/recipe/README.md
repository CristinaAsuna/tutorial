# V-JEPA 真实训练配方合同

此目录记录从 CPU 教学模型走向论文级视频预训练时必须补齐的条件，不是已经验证论文指标的训练脚本。

- 使用固定帧数的视频解码、统一空间裁剪与时序采样；将数据版本写入 checkpoint。
- context encoder 接收 target block 的补集；EMA target encoder 读取完整视频，且从不反传。
- 使用 bf16、DDP/FSDP、断点恢复和 EMA momentum schedule；下游以冻结 backbone 的 action/classification probe 验收。
- 原论文涉及大规模无标签视频及特定评测协议；替换数据集或算力后不能宣称复现原始分数。

运行合同检查：`python validate_config.py`。
