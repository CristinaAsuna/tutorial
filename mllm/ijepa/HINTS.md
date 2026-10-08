# 分层提示

每关先运行检查；按需逐层展开，完整答案在 reference 文件。

## 第 1 关

<details><summary>一级：理解目标</summary>

矩形 target 与 context 补集。先在纸上验证：2×3 网格左上 1×2 target=[T,T,F,F,F,F]，context 逐位取反。
</details>

<details><summary>二级：操作顺序</summary>

1. 校验网格。2. 枚举合法起点。3. 构造矩形并拒绝重叠。4. union 后取补集。每步写出中间 shape。
</details>

<details><summary>三级：定位实现</summary>

必须有 context；采样要有限终止而非无限 while。对照同目录 reference 中对应函数，但只在完成当前微任务后比较；检查器会给出数值与梯度断言。
</details>

## 第 2 关

<details><summary>一级：理解目标</summary>

拼接 context 和位置条件 target slots。先在纸上验证：context=[1,2],mask=10,pos=[3,4] → [1,2,13,14]。
</details>

<details><summary>二级：操作顺序</summary>

1. 对齐 batch 和 D。2. expand mask。3. mask+target pos。4. dim=1 拼接。每步写出中间 shape。
</details>

<details><summary>三级：定位实现</summary>

target slot 不能包含 teacher latent；不要沿 feature 轴拼。对照同目录 reference 中对应函数，但只在完成当前微任务后比较；检查器会给出数值与梯度断言。
</details>

## 第 3 关

<details><summary>一级：理解目标</summary>

更新冻结的 target encoder。先在纸上验证：t=2,s=6,m=.75 → 3；m=1 保持 teacher。
</details>

<details><summary>二级：操作顺序</summary>

1. 校验 momentum。2. no_grad EMA。3. requires_grad False。4. eval。每步写出中间 shape。
</details>

<details><summary>三级：定位实现</summary>

eval 不等于无梯度；不要反向更新 student。对照同目录 reference 中对应函数，但只在完成当前微任务后比较；检查器会给出数值与梯度断言。
</details>

## 第 4 关

<details><summary>一级：理解目标</summary>

完成一次 latent regression 训练。先在纸上验证：zero_grad → forward → backward → step → EMA(0.9)。
</details>

<details><summary>二级：操作顺序</summary>

1. 清梯度。2. 前向得到 loss。3. 反传。4. step。5. EMA。6. 返回 loss。每步写出中间 shape。
</details>

<details><summary>三级：定位实现</summary>

teacher 全图编码后 gather；student 编码前只选可见 token。对照同目录 reference 中对应函数，但只在完成当前微任务后比较；检查器会给出数值与梯度断言。
</details>

## Predictor 骨架如何复用

第 2 关的输入已经是 `in_proj(context)`，最后维度为 predictor_dim。`target_pos_proj` 把 encoder 的位置参数映射到同一维度，再传入 packing 函数；`mask_token` 只有 `[1,1,Dp]`，需要广播到 `[B,M,Dp]`。packing 只做输入组织，不计算 teacher targets。

`practice_ijepa.py` 在拼接后运行已学过的 pre-LN Transformer；所有 target slots 可通过 self-attention 读取 context。`x[:, -M:]` 保留 target slots，LayerNorm 与 `out_proj` 恢复 encoder_dim，所以 Smooth-L1 的 prediction 与 teacher target 都是 `[B,M,D]`。本 toy 把多块合在一次 predictor，正式 I-JEPA 按目标块组织训练；不要把这个简化写成论文全部架构。

## 第 4 关微任务 0：回归 loss

在训练步骤之前独立实现 `latent_regression_loss`。输入都是 `[B,M,D]`，teacher targets 必须 detach；`smooth_l1_loss(..., reduction="mean")` 返回零维标量。practice 模型实际调用本函数，训练步骤使用这个标量；不会把参考 loss 当学生答案。
