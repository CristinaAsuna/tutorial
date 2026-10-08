# 分层提示

先完成关卡局部检查，再打开下一层；三级是伪代码提示，完整答案留在 reference。

## 关卡 1：删除 CLS 后投影

<details><summary>一级：方向</summary>

先标出 token 轴，再区分冻结塔与可训练 projector。

</details>

<details><summary>二级：步骤</summary>

no_grad 只覆盖 vision；tokens[:,1:,:] 才是 patch。

</details>

<details><summary>三级：伪代码</summary>

在冻结区得到 features，退出后 return projector(features[:,1:,:])。

</details>

## 关卡 2：image packing

<details><summary>一级：方向</summary>

同时画 embeddings、mask、labels 三条序列，插入同样长的视觉段。

</details>

<details><summary>二级：步骤</summary>

先找 pos，再 cat 两段文本和 image；只有正 token id 进入 embedding。

</details>

<details><summary>三级：伪代码</summary>

先实现 replace_image_sentinel：分别 embedding 两侧再 cat。再实现 expand_image_supervision：视觉 labels 用 full((N,),-100)，mask 用 ones(N)。外层验证、循环与 pad_sequence 已提供，无须重写。

</details>

## 关卡 3：assistant loss mask

<details><summary>一级：方向</summary>

loss mask 是角色信息，不是 token 值推断。

</details>

<details><summary>二级：步骤</summary>

visible=assistant_mask & attention_mask & (input_ids!=-200)。

</details>

<details><summary>三级：伪代码</summary>

没有显式角色 mask 时 arange(L)>=assistant_start；torch.where 返回 labels，不提前 shift。

</details>

## 关卡 4：训练一步

<details><summary>一级：方向</summary>

冻结控制参数是否更新，不能切断输入的梯度。

</details>

<details><summary>二级：步骤</summary>

顺序 train -> zero_grad -> forward -> backward -> step。

</details>

<details><summary>三级：伪代码</summary>

loss=model(**batch)["loss"]；对 loss.backward()，step 后 return loss.detach()。

</details>
