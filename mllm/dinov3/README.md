# DINOv3：从零实现 Gram anchoring 与稠密特征 refinement

本课对应 **DINOv3（2025）第4节**，先完成 [DINOv2 六关](../dinov2/README.md)。目标是解释并实现局部特征关系的约束，使用上一课的轻量训练底座，**不宣称实现完整 DINOv3 架构或训练配方**。

长训练中全局分类表征与局部 patch 一致性不一定同时变好。论文观察到 dense features 的退化，并用较早的优质教师保存 patch 之间的相似关系。本课保留这个监督与更新机制，不在几步合成训练中伪造长训练退化现象。

## 运行与学习顺序

仓库根、已安装 PyTorch 的 Python：

```bash
PYTHONPATH=general_utils/edu_core python3 -m mllm.dinov3.check_lessons --implementation reference
PYTHONPATH=general_utils/edu_core python3 -m mllm.dinov3.run_dinov3_demo --output-dir /tmp/dinov3-diagnostics
PYTHONPATH=general_utils/edu_core python3 -m mllm.dinov3.check_lessons --lesson 1
```

输出 `metrics.json` 与 `patch_similarity.svg`：相同图案在初始、refinement前、refinement后的 patch cosine 热图。默认8步，至少6步才能经过快照、refinement与刷新。热图和熵/方差是诊断，不能作为论文 dense benchmark 或消除塌缩的证据。

|关卡|机制|手算与局部验收|
|---|---|---|
|1 Gram|normalize(D)，构造B,N,N关系矩阵|正交patch得单位矩阵；旋转后关系不变|
|2 anchoring|student/Gram teacher关系对齐|sum与mean尺度、teacher detach、负相似度截断|
|3 teacher|早期快照与周期刷新|第2次更新保存，第6/8次更新刷新|
|4 resolution|同box双分辨率、feature grid对齐|像素编号裁剪；B,64,D→B,16,D|
|5 refinement|基础loss+Gram、两阶段training|参数梯度、optimizer/EMA/快照顺序|

每关提供形状、手算、编号任务和错误提示。[HINTS.md](HINTS.md) 分三层展开；单关检查只替换该关机制，其他作为已完成前置提供。全部完成后运行：

```bash
PYTHONPATH=general_utils/edu_core python3 -m mllm.dinov3.check_lessons --implementation practice
PYTHONPATH=general_utils/edu_core python3 -m mllm.dinov3.run_dinov3_demo --implementation practice
python3 mllm/check_tutorials.py --paper dino dinov2 dinov3 --implementation reference
```

## 特征、目标与梯度

```text
同一原图区域box
  ├─32² → student(有patch mask) → global patches B,16,D → normalize → Gram B,16,16
  ├─32² → EMA teacher(完整图) → DINO/iBOT targets
  └─64² → Gram teacher(完整图) → B,64,D
          → reshape B,D,8,8 → bicubic至4,4 → B,16,D → normalize → Gram B,16,16

DINO + iBOT + .1*sum(KoLeo per global) + gram_weight*mean(Gram loss per global)
  → student backward/optimizer → EMA teacher → 按日程copy到Gram teacher
```

Gram 只使用两个 global crop，不使用 CLS 或 local crops。默认包括 student masked global 的所有 patch positions，与所选官方 `tokens_used='all'` 路径一致；不会把未遮挡位置误当成唯一监督位置。

令单位化 patch features 为 `X`，`G(X)=X Xᵀ`。公式为 `||G(Xs)-G(Xg)||²_F`。它不直接要求每个 feature 坐标相等：对任意正交矩阵 R，`G(XR)=G(X)`。因此特征可以转动，局部关系仍被保留。

第2关同时支持：

- `reduction='sum'`：公式的Frobenius平方，跨batch直接求和。
- `reduction='mean'`：官方 MSE 风格，默认；相同B,N时等于sum除以 `B*N*N`，改变分辨率时损失权重尺度不同。
- `remove_neg=True`：将双方Gram负相似度截为0，对照公开实现选项；默认False先学习论文公式。可通过 `model.ops.gram_loss=functools.partial(model.ops.gram_loss,remove_neg=True)` 切换。

teacher 分支先 detach，student Gram 保留梯度；Gram teacher 和 EMA teacher 均不加入 optimizer，但 student backbone 的 patch embedding、attention 与 mask token 可收到关系损失梯度。

## 两个 teacher 的时间轴

|已完成optimizer次数|Gram teacher动作|下一步训练|
|---|---|---|
|0–1|未启用|基础DINOv2目标|
|2|从当时EMA backbone复制早期快照|继续基础目标|
|4|保持第2步快照|下一次前向开始加入Gram|
|6|从更新后的EMA复制刷新|后续前向使用新快照|
|8|再次刷新|供继续训练使用|

步数为CPU教学压缩值。论文从较早的teacher获得快照，在约1M迭代后加入Gram，并每10k迭代刷新；高分辨率teacher输入为正常分辨率两倍。见[论文 §4.2–4.3](https://arxiv.org/html/2508.10104v1#S4)。不能把“Gram teacher快照”实现成另一个每步同步的EMA对象。

`training_step` 的 step 是更新前计数，必须等于 `model.completed_steps`。顺序固定为 backward → optimizer → EMA → `after_step(step+1)`；刷新结果供下一次前向。模型state保存计数、快照就绪位、最近刷新步及日程配置；恢复完整训练时还要保存 optimizer、RNG 和数据迭代器。本课测试覆盖相同输入/RNG下恢复后下一次更新一致。

## 为什么必须共用 crop box

独立采样两次随机crop，即使shape相同，patch `(i,j)` 也未必对应同一物体区域。`aligned_crops` 先采样原图box，再分别resize到32和64。高分辨率features先按行优先token次序还原空间网格，再bicubic降采样；**降采样之后**才L2归一化并计算Gram。

当前只用几何增广，因而low/high精确共享区域；完整生产版颜色扰动和高分辨率增强策略需按官方配方接入。

## 复用、来源与边界

`reference_dinov3.py` 提供 `DINOv3` / `build_toy_dinov3()`；practice只替换本课Gram、对齐、teacher策略与训练函数，上一课DINOv2作为已完成前置提供。不是运行答案后覆盖loss，也不会吞掉TODO。

机制分别对应：

- [论文](https://arxiv.org/abs/2508.10104)：§4.1 dense退化观察、§4.2关系约束、§4.3高分辨率教师。
- [官方 GramLoss](https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/gram_loss.py)：归一化、相似度矩阵、MSE、负相似度选项。本课未实现额外的 `remove_only_teacher_neg` 变体。
- [官方训练图](https://github.com/facebookresearch/dinov3/blob/main/dinov3/train/ssl_meta_arch.py)：独立Gram teacher、选取patch、对齐与权重。本课采用图像内Gram，未实现跨batch Gram模式。

沿用的toy是learned二维位置插值、小MLP、单进程KoLeo；没有完整DINOv3的RoPE架构、register tokens、分布式KoLeo、全局loss重加权、多学生蒸馏与文本对齐。它们不能由本课代码推断为已实现。真实训练路线见 [recipe](recipe/README.md)。
