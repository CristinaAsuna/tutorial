# 分层提示

每关先读一级，仍卡住再读二级、三级。三级提供伪代码；具体实现仍在 lesson 中完成。

## 基础 1：像素切片

**一级：** 一个 patch 必须同时包含某块区域的所有通道；网格坐标和 patch 内坐标是不同轴。

**二级：** 高度拆成 `h,p`，宽度拆成 `w,p`。将 C 移到最后，h/w 放到局部 p/p 之前。

**三级：** `reshape(B,C,h,p,w,p) → permute(0,2,4,3,5,1) → reshape(B,h*w,p*p*C)`。逆变换从 `(B,h,w,p,p,C)` 恢复到 `(B,C,h,p,w,p)`，对应轴号 `0,5,1,3,2,4`。

## 基础 2：时空切片

**一级：** 一个 tubelet 是连续 t 帧里的同一块区域；不能逐帧切片后随意拼接。

**二级：** 拆分后有八个轴，网格轴为 `nt,nh,nw`，局部轴为 `t,p,p,C`。

**三级：** 对 `(B,C,nt,t,nh,p,nw,p)` 使用轴号 `0,2,4,6,3,5,7,1`，再合并为 `(B,nt*nh*nw,t*p*p*C)`。

## 基础 3：attention 与残差

**一级：** 每个 query 对 key 分配概率，然后加权读取 value。遮蔽发生在 softmax 前。

**二级：** 分数 `(B,H,Q,K)`；用 dtype 最小值填不可见位置。全遮蔽时 softmax 会错误地产生均匀权重，应额外将该行权重置零。out projection 有 bias，最后仍需清零该行。

**三级：** `scores=Q@Kᵀ/sqrt(d) → masked_fill → softmax(-1) → 清零空行 → weights@V`。投影结果先 reshape 为 `(B,S,H,d)` 再交换 S/H；合头时反向交换。`x←x+attn(norm1(x))` 后，再执行 `x←x+mlp(norm2(x))`。

## 基础 4：mask 与选择

**一级：** 因果关系是 key 的位置不大于 query 的位置。索引选择必须保持每个样本自己的顺序。

**二级：** query 位置放列向量，key 放行向量；gather 的 index 与目标输出应有相同轴数。

**三级：** `arange(L)[None,:] <= arange(L)[:,None]`；`tokens.gather(1, indices[...,None].expand(-1,-1,D))`。

## 基础 5：位置插值

**一级：** CLS 没有二维坐标，单独保留。只有 patch 的网格发生大小变化。

**二级：** 先将 `(1,h*w,D)` reshape 为 `(1,h,w,D)`，再变为 PyTorch 插值需要的 `(1,D,h,w)`。

**三级：** 分离 prefix；`reshape → permute(0,3,1,2) → bicubic interpolate → permute(0,2,3,1) → reshape(1,H*W,D)`；最后沿 token 轴拼回 prefix。

## 基础 6：EMA

**一级：** momentum 越大，teacher 越缓慢跟随 student。m=1 保持 teacher，m=0 完全复制 student。

**二级：** 先完成 student optimizer 更新，再在 no_grad 中逐参数更新 teacher；不要让 loss 更新 teacher。

**三级：** 遍历一一对应的参数，`target.mul_(m).add_(source, alpha=1-m)`。这里只更新参数；DINO center 是 buffer，需要另外的明确更新规则。
