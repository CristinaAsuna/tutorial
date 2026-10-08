"""关卡 4：完成一次 latent regression 训练

前置：前3关、optimizer 顺序。
形状合同：images BCHW，BlockMasks；返回 scalar Tensor。
手算例子：zero_grad → forward → backward → step → EMA(0.9)。
编号 TODO：1. 清梯度。2. 前向得到 loss。3. 反传。4. step。5. EMA。6. 返回 loss。
常见错误：teacher 全图编码后 gather；student 编码前只选可见 token。
检查：python3 check_lessons.py --lesson 4 --implementation practice
提示：HINTS.md 第 4 关；参考检查可加 --implementation reference。
"""
from torch import Tensor


def jepa_training_step(model, images: Tensor, masks, optimizer):
    # TODO 1: optimizer.zero_grad()；2: model(images,masks)["loss"] 得 scalar。
    # TODO 3: loss.backward()；4: optimizer.step()。
    # TODO 5: model.update_target_encoder(0.9)，必须在 step 之后。
    # TODO 6: 返回 scalar loss；检查器还会验证参数/teacher EMA 的实际更新。
    raise NotImplementedError("Implement an I-JEPA training step")


def latent_regression_loss(predictions, targets):
    """微任务 0：[B,M,D] prediction/target → scalar Smooth-L1，target 无梯度。

    手算：prediction=0,target=1 时 beta=1 的 Smooth-L1 为0.5。
    """
    # TODO 0a: 检查两者形状一致；0b: targets.detach()。
    # TODO 0c: torch.nn.functional.smooth_l1_loss 默认 mean，返回 scalar。
    raise NotImplementedError("lesson4.latent_regression_loss TODO 0")
