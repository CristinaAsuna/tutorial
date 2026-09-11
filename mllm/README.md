# Modern MLLM / vision-learning tutorials

每篇目录都有两条明确分开的轨道：顶层 `reference_*.py` 和 `lesson*.py` 是小模型、CPU toy 的机制教学；`recipe/` 是真实数据、外部预训练权重和分布式训练所需的配方骨架。后者不会因为配置文件存在就声称已复现论文指标。

先安装共享基础包：

```bash
python3 -m pip install -e "../general_utils/edu_core[dev]"
```

`general_utils/utils` 继续服务扩散/生成模型。MLLM 使用 `general_utils/edu_core` 提供的 `edu_core` token 语义，避免把空间 GroupNorm attention 或 `[-1,1]` 图像约定混入文本序列训练。
