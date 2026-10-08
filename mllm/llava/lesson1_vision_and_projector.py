"""关卡 1：冻结视觉塔，删除 CLS，投影 patch。
目标：图像变成 LLM 可读的 token；前置：基础 patch embedding、no_grad、Linear。
形状：(B,3,8,8) -> (B,5,24) -> (B,4,24) -> (B,4,32)。
手算：特征 [CLS=99, patch=1, patch=2] 只能留下 [1,2]。
常见错：把 CLS 当图像 patch；把 projector 也放进 no_grad，截断训练。
接入：PracticeLlava.encode_images；检查：check_lessons.py --lesson 1。
"""
import torch


def vision_to_llm_tokens(vision_encoder, projector, pixel_values: torch.Tensor) -> torch.Tensor:
    # TODO 1.1：仅在 no_grad 内运行被冻结的 vision_encoder。
    # TODO 1.2：用 [:, 1:, :] 删除 CLS，保留所有 patch。
    # TODO 1.3：在 no_grad 外调用 projector，让投影参数仍获得梯度。
    raise NotImplementedError("LLaVA lesson 1: vision_to_llm_tokens 尚未完成；见 HINTS.md")
