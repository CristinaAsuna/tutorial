# edu_core

`edu_core` 是 `/scratch` 下教程共享的最小 PyTorch 基础包。它只包含没有论文策略含义的 token attention、ViT patch/tubelet token、位置编码、padding/label helpers、冻结/EMA 与 checkpoint state；不包含数据集、模型训练器、扩散 UNet 或某篇论文的损失。

```bash
python3 -m pip install -e "./general_utils/edu_core[dev]"
python3 -m pytest ./general_utils/edu_core/tests
```

Mask 统一约定为：布尔值 `True` 或数值 `1` 表示该 key/token 有效、可以被 attention 看见。`GatedCrossAttentionBlock` 可把任意外部 token memory 接入冻结序列模型，且其零初始化 gate 使初始输出严格保持原模型路径。各论文目录保留其课程关键计算，而只导入此包的通用组件。

## 基础课与复用接口

基础原理可以先在 [`mllm/foundations`](../../mllm/foundations/) 练习，后续论文直接导入共享实现。

|接口|输入与输出|约定|
|---|---|---|
|`patchify(images,p)`|BCHW → B,N,p²C|可逆；局部顺序 p,p,C，网格宽度最快|
|`unpatchify(patches,p,channels,grid=...)`|B,N,p²C → BCHW|矩形必须传 grid，省略时要求方形网格|
|`tubelet_patchify(videos,t,p)`|BCTHW → B,N,tp²C|网格 T,H,W；局部顺序 t,p,p,C|
|`PatchEmbed` / `TubeletEmbed`|像素 → B,N,D 和 grid|卷积学习投影，不是无损切片|
|`MultiHeadAttention`|B,Q,D 和可选 B,K,Dkv → B,Q,D|mask 为 Q,K 或 B,Q,K；True 表示允许|
|`SelfAttentionBlock`|B,N,D → B,N,D|纯 Pre-LN encoder，无 cross-attention 参数|
|`update_ema(teacher,student,m)`|原地更新 teacher 参数|不更新 buffers；调用方负责参数对应、optimizer 后更新及冻结策略|

旧 `TransformerBlock` 继续保留自注意力加可选交叉注意力接口。共享包不包含论文的 target 采样、teacher 策略或损失；现有 checkpoint 的参数布局不因新增组件自动转换。
