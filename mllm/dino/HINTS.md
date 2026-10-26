# 分层提示

每关先运行检查；按需逐层展开，三级是引导式伪代码，完整答案在 reference 文件。

## 第 1 关

<details><summary>一级：理解目标</summary>

图像变成 CLS 与 patch 表征。先在纸上验证：P=2,H=W=4 时 N=4；加 CLS 后序列长 5。

</details>

<details><summary>二级：操作顺序</summary>

参数与 Transformer 已提供；只完成 mask 替换，以及拼接 CLS 与插值位置。每步写出中间 shape。

</details>

<details><summary>三级：伪代码</summary>

replace_masked_patches：mask 为 None 时原样返回（shape/dtype 不变）；否则用 torch.where 只替换 mask=True 的 patch，mask_token 广播到 patch 维度。add_cls_and_positions：先对 patch 网格插值 2D 位置，再把 CLS 拼到 patch 之前，最后加位置；CLS 不属于 patch mask。

</details>

## 第 2 关

<details><summary>一级：理解目标</summary>

构造两 global、四 local 与定量 mask。先在纸上验证：N=4,ratio=0.5 每行恰好两处 True。

</details>

<details><summary>二级：操作顺序</summary>

随机采样外围已提供且 globals 在前。先实现 crop_given_boxes 确定性裁剪，再实现每行随机选固定数量 patch。

</details>

<details><summary>三级：伪代码</summary>

crop_given_boxes：boxes 每行是原图像素坐标 (top,left,height,width)，不是归一化坐标或 patch 索引。逐样本校验边界后切片 images[i:i+1,:,top:top+h,left:left+w]，bilinear 到 output_size（align_corners=False），收集后沿 batch 拼接。crop scale 是面积比例（.25 即方形边长一半），mask ratio 是遮挡 patch 比例，两者不要混。random_patch_mask：不要把随机性放进确定 helper（采样器已提供 Bernoulli、ratio、搜索与残余填充），对每行固定数量 round(N*ratio) 选 patch。

</details>

## 第 3 关

<details><summary>一级：理解目标</summary>

跨视图软交叉熵与 center。先在纸上验证：2 global+1 local：有效配对 (t0,s1),(t0,s2),(t1,s0),(t1,s2) 共 4；均匀 K=2 loss=log(2)。

</details>

<details><summary>二级：操作顺序</summary>

1. teacher 减 center/温度/detach。2. student log_softmax。3. 排除同 global。4. 平均配对。5. 更新 center。每步写出中间 shape。

</details>

<details><summary>三级：伪代码</summary>

teacher_distribution：softmax((teacher.detach()-center)/teacher_temp, dim=-1)。soft_cross_entropy：logp=log_softmax(student/student_temp, dim=-1)，按类别轴 -(q*logp).sum(-1)。第 3 关每个有效 view pair 先对 batch 平均，再对有效 pair 平均。update_center：在 no_grad 或对 logits detach 后用原始 logits 均值——例 center=0,mean_logits=[1,1],m=.5 → [.5,.5]；更新后的 center 下次前向才使用。

</details>

## 第 4 关

<details><summary>一级：理解目标</summary>

masked patch 蒸馏与 teacher EMA。先在纸上验证：teacher=[1,0], student=[0.5,0.5] 时 CE=log(2)；t=2,s=6,m=.75 更新后 t=3。

</details>

<details><summary>二级：操作顺序</summary>

1. teacher detach/中心化。2. 每 patch CE。3. 仅 mask True 平均。4. no_grad EMA。每步写出中间 shape。

</details>

<details><summary>三级：伪代码</summary>

ibot_masked_patch_loss：复用第 3 关 soft CE 得到 (B,N)，只对 mask=True 的值平均，分母是 masked patch 数量。ema_update：optimizer.step 之后、no_grad 下做 teacher=m*teacher+(1-m)*student（用已更新的 student 参数）；teacher 永不加入 optimizer。

</details>
