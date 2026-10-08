"""关卡 2：用 current proprioception 和 future actions 推断训练期 style latent z。"""


def cvae_loss(predicted_actions, target_actions, action_mask, mean, logvar, beta):
    # TODO: valid-only L1 reconstruction + beta * KL(q(z|a,qpos) || N(0,I))。
    raise NotImplementedError("参考 masked_l1 和 kl_to_standard_normal")
