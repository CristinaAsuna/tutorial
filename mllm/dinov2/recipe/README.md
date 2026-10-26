# 从CPU机制到真实DINOv2实验

这里是元数据配方与实验路线，**不是可执行的分布式训练器**。先通过toy数值/梯度验收，再迁移到[官方训练实现](https://github.com/facebookresearch/dinov2)。配置中的模型、路径和日程必须按实验修改；运行校验不启动训练。

```bash
# 仓库根；缺失数据时仍可检查配置，但明确报告data_exists=false。
python3 -m mllm.dinov2.recipe.validate_config
python3 -m mllm.dinov2.recipe.validate_config --require-data
```

真实数据可以是无标签RGB文件夹；正式训练需适配官方dataset/manifest，记录文件清单hash、许可、筛选/去重版本、训练/评估划分，不能把随机toy图当ImageNet实验。避免评估图像泄漏到训练。

迁移顺序：

1. 保留2 global、多local、teacher完整/student masked与独立heads，接入官方RandomResizedCrop与颜色、灰度、模糊、solarization、归一化；记录每项概率和global/local分辨率。
2. 使用真正的ViT配置、SwiGLU/LayerScale/stochastic depth和论文优化参数；记录温度warmup、teacher momentum、学习率与weight decay曲线、末期高分辨率适配。
3. 按官方FSDP/混合精度训练入口配置资源；SK须使用跨卡全局分配，center对照须同步统计。CPU函数只处理本进程，不能简单放进DDP后称等价。
4. 保存student、teacher、heads、optimizer、scheduler、RNG、sampler位置与数据版本；恢复后检查teacher与student更新顺序。
5. 分别评估冻结backbone的k-NN、linear probe及dense linear probe；记录数据集、输入分辨率、训练head协议和指标。不要只用训练loss代表表征质量。

大规模数据筛选、长日程和硬件条件无法由本目录的小模型自动复现；任何正式结果必须单独记录实际命令、数据与资源。本配置保留`knn/linear_probe/dense_probe`三个评估标签，尚未提供这些真实数据评估器。
