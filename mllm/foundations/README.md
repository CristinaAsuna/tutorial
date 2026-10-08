# 基础算法：练一次，后续论文复用

这里先学习重复出现的张量操作。基础练习不会自动替换共享库：验证后，论文课默认导入完整、经过测试的 `edu_core`，让你专注论文特有机制。

从仓库根目录运行：

```bash
PYTHONPATH=general_utils/edu_core python3 mllm/foundations/check_lessons.py --lesson 1
PYTHONPATH=general_utils/edu_core python3 mllm/foundations/check_lessons.py --implementation reference
```

或者先用当前 Python 安装 `python3 -m pip install -e 'general_utils/edu_core[dev]'`，以后无需设置 `PYTHONPATH`。本机已验证的解释器是 `/Users/max/codebase/.ml/.venv/bin/python`，它只是环境实例，不是课程依赖。

|关卡|亲手实现|论文中使用的组件|
|---|---|---|
|1|2D patchify/unpatchify；先理解像素重排|`edu_core.vision.patchify/unpatchify`；学习投影用 `PatchEmbed`|
|2|3D tubelet patchify；理解时间轴和展平顺序|`tubelet_patchify`；学习投影用 `TubeletEmbed`|
|3|缩放点积、遮蔽、多头与 Pre-LN 残差|`MultiHeadAttention`、`SelfAttentionBlock`|
|4|因果 allow-mask 与按 batch 选择 token|`causal_allow_mask`；论文自己确定要选哪些位置|
|5|CLS 分离与二维位置网格插值|`interpolate_2d_pos_embed`|
|6|已知参数上的 EMA 数值更新|`update_ema`；论文自己管理 teacher frozen/eval|

推荐先完成 1、3、4、6，然后进入 DINO 或 LLaVA；DINO 补 5，V-JEPA 补 2。每关有独立检查，未完成只报告对应 TODO；不必先写完全部基础再开始论文课。

`True` 在 attention mask 中表示允许看见；在论文的 target mask 中可能表示选择目标，务必按变量名区分。2D 网格按高度、宽度排列，宽度变化最快；局部像素按 `(p,p,C)` 排列。3D 网格按 `(T,H,W)`，局部像素按 `(t,p,p,C)` 排列。

先尝试，再按需要阅读 [分层提示](HINTS.md)。检查使用小数值、内容顺序和信息隔离，而不只是张量形状；参考模式通过只说明答案及检查器可运行。
