# 三级提示：先尝试，再逐层展开

每关答案独立检查；不要把“参考通过”记为练习完成。

## 第1关：矩形与mask数量

<details><summary>一级：为什么</summary>

先在纸上画2×3网格，标出top=0,left=1,height=2,width=1覆盖的两格。是否选择遮挡、遮挡数量与选择哪几格是三个问题，先解决前两个。

</details>

<details><summary>二级：张量操作</summary>

rectangle_mask创建bool网格，只改一个二维切片；mask_count在selected=False时返回0，其余round后至少1。随机搜索外围会把你的矩形函数用于block union。

</details>

<details><summary>三级：伪代码</summary>

rectangle_mask 建 H×W 的 bool 网格，只把 [top:top+h, left:left+w] 这一个二维切片置 True。mask_count 未选中返回 0，否则 round(N*ratio) 且至少 1。不要把随机性放进确定 helper——采样器已提供 Bernoulli、ratio、搜索次数和残余填充。

</details>

## 第2关：独立heads与teacher

<details><summary>一级：为什么</summary>

独立是参数对象独立，不只是两个不同变量名；两个目标可以有相同shape和初始数值。teacher既不能有梯度，也不能处于train模式。

</details>

<details><summary>二级：张量操作</summary>

choose_patch_head直接选择已初始化的对象；initialize_teacher先load_state_dict，再requires_grad_(False)与eval。只操作teacher。

</details>

<details><summary>三级：伪代码</summary>

choose_patch_head 在 shared 时返回 cls_head，否则返回候选 patch head。initialize_teacher 先把 student 的 state 拷进 teacher，再对 teacher 设 requires_grad_(False) 与 eval()。构造后两个 head 的参数对象要独立（id 不相交）；teacher 与 student 数值相等但 storage 不同。

</details>

## 第3关：Sinkhorn行列边际

<details><summary>一级：为什么</summary>

Q是K×N，行代表prototype、列代表样本。总质量先归一到1，每行1/K、每列1/N。最后乘N，才成为每个样本和为1的target。

</details>

<details><summary>二级：张量操作</summary>

先detach并转换float64；在log-space中归一化就是减logsumexp和目标质量的负log。循环默认三次，顺序是行再列。

</details>

<details><summary>三级：伪代码</summary>

在 no_grad 下：先 detach 转 float64，除以温度后转置成 K×N。log 空间归一化即「减 logsumexp 与目标质量的 -log」——行目标质量 -log(K)、列 -log(N)。按 行→列 顺序各减一次 logsumexp，循环默认三次（只保证行边际近似）。最后乘 N 转回原 dtype。

</details>

## 第4关：逐样本iBOT

<details><summary>一级：为什么</summary>

每个样本应有相同权重，不应因为某个图mask多就占更多loss。零mask样本仍在batch里，只贡献0。teacher已经给概率，不要再次softmax。

</details>

<details><summary>二级：张量操作</summary>

类别轴计算CE得到B,N；乘patch_mask，sum到B；分母是每行mask数量clamp_min(1)，最后mean。

</details>

<details><summary>三级：伪代码</summary>

类别轴 CE 得到 (B,N)：teacher 概率先 detach，student 走 log_softmax。乘 patch_mask 后沿 patch 轴求和，分母是每行 mask 数量 clamp_min(1)，最后对 batch mean。零 mask 样本保留计算图、只贡献 0，不要先 boolean gather 再全局 mean。

</details>

## 第5关：KoLeo近邻

<details><summary>一级：为什么</summary>

单位化后最大cosine就是最近欧氏距离。最近邻必须是另一个样本。索引离散，可以停止梯度；最终距离必须从原始可微特征计算。

</details>

<details><summary>二级：张量操作</summary>

normalize(features.float(),dim=-1)，相似度对角设-inf，argmax得到B个索引。再求x与x[index]的L2距离，截下限、取-log平均。

</details>

<details><summary>三级：伪代码</summary>

在 no_grad 下：单位化特征（dim=-1），算相似度矩阵并把对角设 -inf 后 argmax 得到每行最近邻索引。距离必须从原始可微特征算 x 与 x[index] 的 L2，截下限后取 -log 再 mean。要求 B>=2；两个 global 分别算再按训练权重合并。

</details>

## 第6关：联合训练与日程

<details><summary>一级：为什么</summary>

teacher是过去student的平滑副本。先EMA再optimizer会让teacher落后一次更新。温度是target分布尺度，EMA momentum是参数更新比例，两者不要混淆。

</details>

<details><summary>二级：张量操作</summary>

温度alpha=min(step/warmup,1)，从start到end；warmup=0直接end。EMA调用共享cosine helper。训练清梯度→前向→反向→optimizer→EMA→日志detach。

</details>

<details><summary>三级：伪代码</summary>

顺序：model.train() → 清梯度(set_to_none=True) → out=model.losses(crops,masks,step) → out["loss"].backward() → optimizer.step() → model.update_teacher(ema_momentum(step,total_steps))。日志 detach 返回。函数自身不能去调用 reference 的 training_step。

</details>
