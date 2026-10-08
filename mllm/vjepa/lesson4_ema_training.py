"""关卡 4：latent regression 的一次训练

前置：前三关、EMA。
形状合同：videos BCTHW, masks [B,N]；返回 scalar Tensor。
手算例子：optimizer 后 EMA；teacher param m=.9 时新值=.9*t+.1*s。
编号 TODO：1. 清梯度。2. 前向。3. loss backward。4. step。5. teacher EMA。6. 返回 loss。
常见错误：teacher full-video 输出必须 stop-gradient；eval 和 freeze 均需保持。
检查：python3 check_lessons.py --lesson 4 --implementation practice
提示：HINTS.md 第 4 关；参考检查可加 --implementation reference。
"""
import torch


def vjepa_training_step(model, videos: torch.Tensor, target_mask: torch.Tensor, context_mask: torch.Tensor,
                        optimizer: torch.optim.Optimizer, momentum: float) -> torch.Tensor:
    # TODO 1: optimizer.zero_grad()；2: model(videos,target_mask,context_mask)["loss"]。
    # TODO 3: loss.backward()，model forward 已对 teacher stop-gradient。
    # TODO 4: optimizer.step()；5: model.update_target(momentum)。
    # TODO 6: 返回 scalar loss；检查会验证 EMA 数值与 student 更新。
    raise NotImplementedError("TODO: 实现 V-JEPA 单步训练")


def latent_regression_loss(predictions, targets):
    """微任务 0：[B,M,D] prediction/target → scalar Smooth-L1，target 无梯度。

    手算：prediction=0,target=1 时 beta=1 的 Smooth-L1 为0.5。
    """
    # TODO 0a: 检查两者形状一致；0b: targets.detach()。
    # TODO 0c: torch.nn.functional.smooth_l1_loss 默认 mean，返回 scalar。
    raise NotImplementedError("lesson4.latent_regression_loss TODO 0")
