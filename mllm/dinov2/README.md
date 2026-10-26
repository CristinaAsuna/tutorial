# 从 DINO+iBOT 到 DINOv2：六关算法进阶

本课对应 **DINOv2: Learning Robust Visual Features without Supervision（2023）**。先完成 [DINO+iBOT 入门](../dino/README.md) 与 foundations 的 attention、mask、EMA。这里练新增的训练机制，不重复手写基础 ViT。

参考解是一层 ViT 的 CPU 小模型，**不是完整 DINOv2 架构或论文指标复现**。默认无需数据下载、torchvision 或预训练权重。中文练习保持未填写；参考成功不代表练习完成。

## 先运行，再从第一关开始

从仓库根、使用已安装 PyTorch 的 Python 执行：

```bash
PYTHONPATH=general_utils/edu_core python3 -m mllm.dinov2.check_lessons --implementation reference
PYTHONPATH=general_utils/edu_core python3 -m mllm.dinov2.run_dinov2_demo --output-dir /tmp/dinov2-diagnostics
PYTHONPATH=general_utils/edu_core python3 -m mllm.dinov2.check_lessons --lesson 1
```

`/tmp/dinov2-diagnostics/metrics.json` 保存熵、方差和损失；`patch_similarity.svg` 可在浏览器打开，比较同一固定图案训练前后的 patch 关系。默认 8 步，数据是圆形、背景与条纹组合。它用于观察机制，不要求 loss 单调下降，也不证明消除了表征塌缩。

|关卡|你实际实现什么|接入位置|
|---|---|---|
|1 masking|确定矩形、是否遮挡与数量；随机搜索外围已提供|`model.views` 的 block-union sampler|
|2 heads|选择独立/共享 head；复制并冻结 teacher|`scaffolding.build_modules`|
|3 Sinkhorn|log-space 行列归一化、teacher stop-grad|CLS 和选中 teacher patch 的 targets|
|4 iBOT|每样本按 masked 数归约，再对全 batch 平均|联合训练 loss|
|5 KoLeo|排除自己、选择近邻、可微距离|-log 最近邻距离|
|6 training|温度/EMA日程、反传与更新顺序|实际 CPU training_step|

每关有手算与 shape；卡住看 [HINTS.md](HINTS.md)，按思路 → 操作 → 伪代码展开。每个 `--lesson N` 独立运行，不要求后续 TODO 完成。全部完成后：

```bash
PYTHONPATH=general_utils/edu_core python3 -m mllm.dinov2.check_lessons --implementation practice
PYTHONPATH=general_utils/edu_core python3 -m mllm.dinov2.run_dinov2_demo --implementation practice
python3 mllm/check_tutorials.py --paper dino dinov2 --implementation reference
```

## 输入 → 表征 → 监督 → 更新

```text
RGB images B,3,80,80
  → 每张图独立采样2 global(32²)、4 local(16²)
  → student: global随机block mask，local完整
       backbone CLS B,32 / patches B,N,32
       独立 dino_head 与 ibot_head → K=16 prototype logits
  → teacher: 同一global，不遮挡；无梯度
       CLS 拼成2B,K → Sinkhorn → 2个B,K target
       patches按student mask选中 → Sinkhorn → 放回2B,N,K
  → DINO跨视图 + iBOT同位置 + .1 × 两个global各自KoLeo之和
  → backward → optimizer.student → EMA.teacher
```

`P=8` 时 global 网格 `4×4`，local 网格 `2×2`。`patch_mask=True` 表示遮挡输入像素对应的 token；它与 attention 的 `allow_mask=True`（允许看见）语义不同。遮挡不是删 token：位置保持不变，teacher/student 可逐位置匹配。

### 为什么需要这三个增量

**独立 head。** DINO 关心图像级 CLS，iBOT 关心局部 patch。原 iBOT 曾使用共享 head；DINOv2 论文在规模化实验中分开两组参数。`build_toy_dinov2(shared_head=True)` 可作共享对照；默认两个 head 的参数身份不相交。teacher 的两组参数分别从 student 初始化，再由 EMA 更新。

**Sinkhorn-Knopp。** 普通 softmax 只保证每个样本概率和为1。SK 还推动 batch 中 prototype 使用量平衡。令 `Q=exp(logits/T).T`，形状 `K×N`，交替把每行总质量归一到 `1/K`、每列到 `1/N`，最后返回 `N*Q.T`。实现用 float64 log-space 防止溢出。默认三轮：最后的样本概率和精确为1，prototype边际只近似平衡；测试用100轮验证收敛，不能要求任意尖锐输入三轮后精确平衡。

SK 用于 teacher；student 仍是温度为 `.1` 的 log_softmax。只有被选择为 iBOT 目标的 patch 参与 patch SK，未遮挡位置不混入其边际。`centering='center'` 提供传统移动中心对照；中心作为 buffer 更新，下一次 target 使用新中心。

**按样本 iBOT。** 设第 b 行有 `m_b` 个 mask，先算 `sum(mask*CE)/max(m_b,1)`，再对**全部**行平均。例：两行 mask 数1、3，CE总和2、12，结果 `(2+4)/2=3`，不是 `14/4`。零 mask 行贡献0，仍计入总行数；全批零 mask 的 loss 是可反传的0。

**KoLeo。** 对 backbone CLS 做单位化，按最大 cosine similarity 选最近的其他样本，再对欧氏距离算 `-mean(log(distance))`。选索引不需要梯度，计算距离需要梯度。两个 global 分开做，不把同一图另一裁剪放进近邻池。两正交向量给出 `-log(sqrt(2))`，loss 可以为负。CPU 用距离下限 `1e-8` 避免重复特征的 log(0)；该数值细节与官方 PairwiseDistance 的 eps 形式不同。

### 日程与顺序

温度 `.04→.07`，默认 warmup=4：step `0/2/4` 分别 `.04/.055/.07`。EMA momentum 从 `.994` cosine 升至1。step 表示本次更新前已完成的次数；8步训练调用 `0..7`，函数在step=8的理论端点为1。

teacher 始终 frozen/eval，optimizer 只持有 student。每次先反传更新 student，再 EMA teacher。共享 EMA 只更新参数；本模型无需同步 BatchNorm 统计。

## 文件与复用接口

- `reference_dinov2.py`：完整答案、`DINOv2` 和 `build_toy_dinov2()`（返回模型，criterion集成在 `model.losses`）。
- `lesson1..6`、`scaffolding.py`、`practice_dinov2.py`：初始化外围与函数注入。practice 不捕获 TODO 后回退答案。
- `check_lessons.py`、`tests/`：数值验收、错误实现反例、维护用练习替身接线；替身不写回学生文件。
- `recipe/`：真实数据/分布式/评估的路径与元数据校验。

通用 PatchEmbed/encoder/插值/EMA 来自 `edu_core`。DINO head、简化 ViT 组装和几何 crop 来自入门课。论文 loss、采样与 teacher 策略不进入通用基础包。新课程用 `python3 -m ...` 从根运行，不修改 sys.path。

## 对照原论文与公开代码

|来源|保留机制|本课边界|
|---|---|---|
|[论文 §4](https://arxiv.org/html/2304.07193v2#S4)|独立head、SK、KoLeo、iBOT|无数据筛选与规模化指标|
|[官方训练图](https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/ssl_meta_arch.py)|teacher完整global、student masked global、分目标更新|按view循环，非packed/FlashAttention|
|[官方iBOT loss](https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py)|逐行mask权重、全batch归约|单进程SK，无跨卡assignments|
|[官方block masking](https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/masking.py)|矩形区域联合|独立Bernoulli/ratio；有界搜索后随机补足精确数量，不复制官方collator分层采样|

仅保留几何增广，缺少完整颜色/灰度/模糊/solarization；未实现真实ViT-g、SwiGLU、LayerScale、stochastic depth、训练末高分辨率适配和分布式优化。它们在 recipe 中明确列为真实训练工作，不能因配置存在声称已实现。

下一课：[DINOv3 Gram anchoring](../dinov3/README.md)。
