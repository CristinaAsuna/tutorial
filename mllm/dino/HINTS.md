# 分层提示

每关先运行检查；按需逐层展开，完整答案在 reference 文件。

## 第 1 关

<details><summary>一级：理解目标</summary>

图像变成 CLS 与 patch 表征。先在纸上验证：P=2,H=W=4 时 N=4；加 CLS 后序列长 5。
</details>

<details><summary>二级：操作顺序</summary>

1. 建立 patch/CLS/mask/position 参数。2. mask 替换。3. 插值位置。4. Transformer 和输出切片。每步写出中间 shape。
</details>

<details><summary>三级：定位实现</summary>

CLS 不属于 patch mask；先插值 patch 网格再拼 CLS 位置。对照同目录 reference 中对应函数，但只在完成当前微任务后比较；检查器会给出数值与梯度断言。
</details>

## 第 2 关

<details><summary>一级：理解目标</summary>

构造两 global、四 local 与定量 mask。先在纸上验证：N=4,ratio=0.5 每行恰好两处 True。
</details>

<details><summary>二级：操作顺序</summary>

1. 保持 globals 在前。2. 采样裁剪并 resize。3. 每行随机选固定数量 patch。每步写出中间 shape。
</details>

<details><summary>三级：定位实现</summary>

不能把 local crop 当 teacher global；ratio 是本 toy 边长比例。对照同目录 reference 中对应函数，但只在完成当前微任务后比较；检查器会给出数值与梯度断言。
</details>

## 第 3 关

<details><summary>一级：理解目标</summary>

跨视图软交叉熵与 center。先在纸上验证：2 global+1 local：有效配对 (t0,s1),(t0,s2),(t1,s0),(t1,s2) 共4；均匀 K=2 loss=log(2)。
</details>

<details><summary>二级：操作顺序</summary>

1. teacher 减 center/温度/detach。2. student log_softmax。3. 排除同 global。4. 平均配对。5. 更新 center。每步写出中间 shape。
</details>

<details><summary>三级：定位实现</summary>

只跳过两个同 global 对；center 取 logits 均值而非概率。对照同目录 reference 中对应函数，但只在完成当前微任务后比较；检查器会给出数值与梯度断言。
</details>

## 第 4 关

<details><summary>一级：理解目标</summary>

masked patch 蒸馏与 teacher EMA。先在纸上验证：teacher=[1,0], student=[0.5,0.5] 时 CE=log(2)；t=2,s=6,m=.75 更新后 t=3。
</details>

<details><summary>二级：操作顺序</summary>

1. teacher detach/中心化。2. 每 patch CE。3. 仅 mask True 平均。4. no_grad EMA。每步写出中间 shape。
</details>

<details><summary>三级：定位实现</summary>

分母是 masked patch 数量；optimizer 后才 EMA。对照同目录 reference 中对应函数，但只在完成当前微任务后比较；检查器会给出数值与梯度断言。
</details>

## 第 3、4 关的数值提示

先算 `q = softmax((teacher.detach()-center)/teacher_temp, dim=-1)`，再算 `logp = log_softmax(student/student_temp, dim=-1)`；按类别轴求 `-(q*logp).sum(-1)`。第 3 关每个有效 view pair 先对 batch 平均，再对有效 pair 平均。第 4 关先得到 `[B,N]`，只挑 `mask=True` 的值平均。

center 更新必须在 `torch.no_grad()` 下或对 logits detach 后做，示例 `center=0,mean_logits=[1,1],m=.5` 得 `[.5,.5]`。center 更新用原始 logits；更新后的 center 从下一次前向开始使用。EMA 使用已更新后的 student 参数，teacher 永远不应加入 optimizer。
