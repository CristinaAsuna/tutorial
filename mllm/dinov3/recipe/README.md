# DINOv3 Gram refinement真实实验路线

本课代码仅实现第4节监督机制；完整DINOv3应以[官方仓库](https://github.com/facebookresearch/dinov3)及[论文](https://arxiv.org/abs/2508.10104)为准，核对代码和权重的许可及访问要求。此处不自动下载或申请权重。

```bash
python3 -m mllm.dinov3.recipe.validate_config
python3 -m mllm.dinov3.recipe.validate_config --require-data
```

校验输出分别记录配置有效、数据存在、训练执行、指标复现；后两项始终false，因为该命令只是metadata检查。

真实路线：

1. 先按[DINOv2真实实验路线](../../dinov2/recipe/README.md)建立数据版本、冻结backbone评估与可恢复训练；完整DINOv3还需其实际架构、位置编码、分布式KoLeo和训练目标权重，不能直接把toy模型改名。
2. 在较早迭代保存具有较好dense表现的teacher checkpoint；用固定的验证集和patch cosine图辅助观察，不能仅按CLS分类结果选择快照。
3. 继续基础训练，到refinement阶段加载早期Gram teacher，保留基础损失并加入Gram；周期刷新来自当前EMA teacher。配置采用论文级示例snapshot=200k、start=1M、interval=10k，必须与所选正式训练配置核对。
4. 高分辨率teacher读取相同crop区域的2倍分辨率图像，feature bicubic降采样至student网格；记录输入增强、抗锯齿、tokens_used、负相似度处理、sum/mean与loss weight，不能混用权重尺度。
5. dense验证使用冻结backbone的分割/深度等正式协议，同时报告全局任务结果；比较无Gram、普通Gram、高分辨率Gram，控制训练预算、数据和初始化。记录局部特征一致性与任务指标，不能由合成热图推断mIoU提升。

高分辨率适配、模型族蒸馏、文本对齐不是本课已实现能力。配置不启动训练，尚无真实数据评估器或分布式runner。论文中的长训练dense退化与修复需真实实验验证。
