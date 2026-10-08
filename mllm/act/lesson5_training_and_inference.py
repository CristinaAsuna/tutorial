"""关卡 5：把 posterior 和 policy 接起来。
目标：训练用随机 posterior z，推理用 z=0；损失仍能回传到两条支路。
前置：已完成 1–4；本关是 glue，不再重新实现前序机制。
形状：forward 输出 actions(B,K,A)、标量损失和 mean/logvar(B,Z)。
手算：没有 action_chunks 时 z 是全零、loss=None；给相同 posterior noise 得相同训练预测。
常见错：对 policy forward 使用 no_grad；推理仍采样；optimizer 清梯度顺序错误。
接入：practice Policy.forward 与 demo 的训练步骤调用本关，不使用 reference forward。
"""
import torch
from lesson2_cvae_style import reparameterize, cvae_loss

def policy_forward(policy, images, qpos, action_chunks=None, action_mask=None, *, beta=10.):
    # TODO 5.1：无 action_chunks：zeros(B,Z)，_decode，返回 actions/loss/reconstruction_loss/kl_loss（损失均None）。
    # TODO 5.2：训练：检查 target shape/mask；style_encoder -> reparameterize -> _decode。
    # TODO 5.3：cvae_loss，返回参考接口字典（含 mean,logvar）；保持张量的计算图。
    raise NotImplementedError("ACT TODO 5.1-5.3: policy_forward")

def training_step(model, optimizer, images, qpos, chunks, valid, beta=.01):
    # TODO 5.4：zero_grad -> model(...,beta=beta) -> loss.backward -> step；返回输出字典。
    raise NotImplementedError("ACT TODO 5.4: training_step")
