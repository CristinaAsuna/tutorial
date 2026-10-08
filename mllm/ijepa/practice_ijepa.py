"""保留 encoder/forward 骨架，实际插槽拼接、mask、EMA、训练均来自练习。"""
import torch
import lesson1_block_masks as l1
import lesson2_predictor_packing as l2
import lesson3_ema_teacher as l3
import lesson4_training_probe as l4
from reference_ijepa import IJEPA as BaseModel, JEPApredictor, BlockMasks

class PracticePredictor(JEPApredictor):
    def forward(self, context, target_pos):
        x = l2.pack_predictor_tokens(self.in_proj(context), self.mask_token, target_pos)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(self.norm(x[:, -target_pos.shape[1]:]))

class IJEPA(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predictor = PracticePredictor(kwargs.get('dim',64), kwargs.get('predictor_dim',96), kwargs.get('predictor_depth',2), kwargs.get('heads',4))
    def latent_regression_loss(self, predictions, targets):
        return l4.latent_regression_loss(predictions, targets)

    def update_target_encoder(self, momentum):
        l3.update_target(self.target_encoder, self.context_encoder, momentum)

def sample_block_masks(grid, *, num_targets=2, block_size=(2,2), generator=None):
    # 显式兼容旧签名：局部 RNG 上下文保证传入 generator 不影响全局随机流。
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("practice masks require a CPU torch.Generator")
    if generator is None:
        targets, context = l1.sample_block_masks(grid, num_targets, block_size)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(generator.get_state())
            targets, context = l1.sample_block_masks(grid, num_targets, block_size)
            generator.set_state(torch.get_rng_state())
    return BlockMasks(tuple(targets), context)


def build_toy_ijepa():
    """返回 IJEPA 小模型，forward 返回含学生 scalar loss 的 dict。"""
    return IJEPA(image_size=32,patch_size=8,dim=48,depth=1,heads=4,predictor_dim=64,predictor_depth=1)
