# 三级提示：先尝试，再逐层展开

每关答案独立检查；不要把“参考通过”记为练习完成。

## 第1关：patch关系矩阵

<details><summary>一级：为什么</summary>

N个patch互相比，会产生N×N的关系，不是D×D。先除去每个patch长度，关系才表示方向相似度。

</details>

<details><summary>二级：张量操作</summary>

features.float()沿最后一维normalize；将最后两轴转置后matmul。student路径不要detach。

</details>

<details><summary>三级：伪代码</summary>

单位化最后维后，把最后两轴转置与自身 matmul，得到 N×N 关系矩阵。student 路径不 detach。两个正交 patch 得 I；整体乘同一正交 R 仍得 I。

</details>

## 第2关：Gram anchoring损失

<details><summary>一级：为什么</summary>

要保留的是patch关系，feature坐标本身可以改变。teacher只提供目标，teacher不参与反传。关系矩阵的mean与Frobenius求和相差元素数量。

</details>

<details><summary>二级：张量操作</summary>

调用传入gram_fn两次，teacher先detach。可选对双方Gram clamp_min(0)，再差平方；按mean或sum返回。

</details>

<details><summary>三级：伪代码</summary>

用传入 gram_fn 分别算 student 与 teacher（后者先 detach）的 Gram；可选对双方 clamp_min(0) 后做差平方，按 mean 或 sum 归约。Ds 与 Dt 可不同，因为比较的是 N×N。

</details>

## 第3关：独立Gram teacher

<details><summary>一级：为什么</summary>

EMA teacher每步变，Gram teacher在一段时间内固定。保存一个Python引用没有保存快照；必须复制state到独立模块。

</details>

<details><summary>二级：张量操作</summary>

completed_steps==snapshot_step时首次copy；completed_steps>refine_start且差值被refresh_every整除时刷新。copy后冻结、eval，EMA源保持不变。

</details>

<details><summary>三级：伪代码</summary>

按 completed_steps 判断：等于 snapshot_step 返回 snapshot；超过 refine_start 且差被 refresh_every 整除返回 refresh；否则 None。sync 在 no_grad 下 load_state_dict(source.state_dict())，再 requires_grad_(False) 与 eval()；EMA 源保持不变。after_step 外围负责 ready 与计数 buffer。

</details>

## 第4关：同box与网格对齐

<details><summary>一级：为什么</summary>

同shape不等于同语义位置。先确定原图box，再resize出low/high。教师高分辨率输出对应更密的patch网格，必须先在空间上降采样。

</details>

<details><summary>二级：张量操作</summary>

aligned_crops两次调用共享crop_and_resize且boxes完全相同。features先B,N,D→B,D,Hs,Ws，再bicubic到Ht,Wt，最后flatten空间、transpose。

</details>

<details><summary>三级：伪代码</summary>

两次 crop_and_resize 用完全相同 boxes 得到 low/high。features 先 B,N,D → B,D,Hs,Ws，再 bicubic 到目标网格（align_corners=False, antialias=False），最后 flatten 空间再 transpose 回 token 轴。归一化交给下一步 Gram，不在 resize 前做。

</details>

## 第5关：refinement闭环

<details><summary>一级：为什么</summary>

Gram是新增约束，不替代基础目标。两个global各自比较，再平均。阶段判断已在model.losses提供；你的training_step必须实际运行此loss和after_step。

</details>

<details><summary>二级：张量操作</summary>

refinement_loss返回total和anchor；training先验证step==completed_steps，backward→optimizer→EMA→after_step(step+1)。日志detach不影响已经完成的backward。

</details>

<details><summary>三级：伪代码</summary>

refinement_loss 返回 (total, anchor)：anchor 是对应视图 Gram 差的 mean，total=base+weight*anchor。training_step 先验证 step==completed_steps，再 backward → optimizer → 更新 student → EMA → after_step(step+1)。禁止每步对 Gram teacher 做 EMA。

</details>
