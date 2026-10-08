# 分层提示

每关先运行检查；按需逐层展开，完整答案在 reference 文件。

## 第 1 关

<details><summary>一级：理解目标</summary>

视频按 tubelet 展平。先在纸上验证：[1,1,2,2,2] 的 arange(8),t=2,p=1 → [[0,4],[1,5],[2,6],[3,7]]。
</details>

<details><summary>二级：操作顺序</summary>

1. 校验整除。2. 拆开网格轴和块内轴。3. 把网格轴移前。4. flatten。每步写出中间 shape。
</details>

<details><summary>三级：定位实现</summary>

token 顺序 time,height,width；块内顺序 channel,time,height,width。对照同目录 reference 中对应函数，但只在完成当前微任务后比较；检查器会给出数值与梯度断言。
</details>

## 第 2 关

<details><summary>一级：理解目标</summary>

生成时空 cuboid 与补集。先在纸上验证：(Tg,Hg,Wg)=(2,2,2)，坐标(1,0,1) flatten index=5。
</details>

<details><summary>二级：操作顺序</summary>

1. 校验 block。2. 每样本放置 num_targets 块。3. 拒绝重叠。4. flatten/取补集。每步写出中间 shape。
</details>

<details><summary>三级：定位实现</summary>

各样本 masked 数必须相同；不能满遮挡。对照同目录 reference 中对应函数，但只在完成当前微任务后比较；检查器会给出数值与梯度断言。
</details>

## 第 3 关

<details><summary>一级：理解目标</summary>

context 和目标位置预测 latent。先在纸上验证：C=3,M=2：拼接长5，输出只取最后2个。
</details>

<details><summary>二级：操作顺序</summary>

1. 建立投影和 mask token。2. context 投影。3. target position 投影+mask。4. 拼接/blocks。5. norm/取最后M/输出投影。每步写出中间 shape。
</details>

<details><summary>三级：定位实现</summary>

predictor_dim 可不同于 D；不可把完整视频 token 交给 student。对照同目录 reference 中对应函数，但只在完成当前微任务后比较；检查器会给出数值与梯度断言。
</details>

## 第 4 关

<details><summary>一级：理解目标</summary>

latent regression 的一次训练。先在纸上验证：optimizer 后 EMA；teacher param m=.9 时新值=.9*t+.1*s。
</details>

<details><summary>二级：操作顺序</summary>

1. 清梯度。2. 前向。3. loss backward。4. step。5. teacher EMA。6. 返回 loss。每步写出中间 shape。
</details>

<details><summary>三级：定位实现</summary>

teacher full-video 输出必须 stop-gradient；eval 和 freeze 均需保持。对照同目录 reference 中对应函数，但只在完成当前微任务后比较；检查器会给出数值与梯度断言。
</details>

## 第 1 关的轴顺序提示

拆分后形状为 `[B,C,Tg,t,Hg,p,Wg,p]`。网格轴在前，块内轴在后：`[B,Tg,Hg,Wg,t,p,p,C]`。先用文档里的八个数验证，再用 checker 的非方形、多通道输入确认所有轴；仅 shape 相同无法证明内容相同。

## 第 4 关的边界

`model(...)` 已提供 full-video teacher、stop-gradient 与 Smooth-L1 损失骨架。本关负责训练顺序，通过 `model.update_target(momentum)` 复用共享 EMA。不能为了通过当前关卡而在训练函数内新建 reference 模型，否则最终 practice demo 会检查错误的参数。

## 第 4 关微任务 0：回归 loss

在训练步骤之前独立实现 `latent_regression_loss`。输入都是 `[B,M,D]`，teacher targets 必须 detach；`smooth_l1_loss(..., reduction="mean")` 返回零维标量。practice 模型实际调用本函数，训练步骤使用这个标量；不会把参考 loss 当学生答案。
