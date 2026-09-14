# edu_core

`edu_core` 是 `/scratch` 下教程共享的最小 PyTorch 基础包。它只包含没有论文策略含义的 token attention、ViT patch/tubelet token、位置编码、padding/label helpers、冻结/EMA 与 checkpoint state；不包含数据集、模型训练器、扩散 UNet 或某篇论文的损失。

```bash
python3 -m pip install -e "./general_utils/edu_core[dev]"
python3 -m pytest ./general_utils/edu_core/tests
```

Mask 统一约定为：布尔值 `True` 或数值 `1` 表示该 key/token 有效、可以被 attention 看见。各论文目录保留其课程关键计算，而只导入此包的通用组件。
