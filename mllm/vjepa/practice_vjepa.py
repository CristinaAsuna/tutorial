"""Conv3d 权重骨架复用；原始 tubelet、mask、predictor 均接学生函数。"""
import torch
import torch.nn.functional as F
import lesson1_tubelet_embed as l1
import lesson2_spatiotemporal_masks as l2
from lesson3_latent_predictor import Predictor as LatentPredictor
from reference_vjepa import MiniVideoViT as BaseEncoder, VJEPA as BaseModel
import lesson4_ema_training as l4

class MiniVideoViT(BaseEncoder):
    def tokens(self, videos):
        embed = self.tubelet_embed
        patches = l1.tubelet_patchify(videos, embed.tubelet_size, embed.patch_size)
        # Flatten Conv3d kernel in the same t,h,w,C order as the exercise.
        tokens = F.linear(patches, embed.proj.weight.permute(0,2,3,4,1).flatten(1), embed.proj.bias)
        grid = (videos.shape[2]//embed.tubelet_size, videos.shape[3]//embed.patch_size, videos.shape[4]//embed.patch_size)
        if grid != self.base_grid:
            raise ValueError('video grid differs from configured base_grid')
        return tokens + self.pos_embed, grid

def sample_spatiotemporal_masks(batch_size, grid, target_block=(1,2,2), num_targets=2, *, generator=None, device=None):
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("practice masks require a CPU torch.Generator")
    if generator is None:
        target, context = l2.make_masks(batch_size, grid, target_block, num_targets)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(generator.get_state())
            target, context = l2.make_masks(batch_size, grid, target_block, num_targets)
            generator.set_state(torch.get_rng_state())
    return target.to(device=device), context.to(device=device)


class VJEPA(BaseModel):
    def latent_regression_loss(self, predictions, targets):
        return l4.latent_regression_loss(predictions, targets)


def build_toy_vjepa():
    """返回 VJEPA 小模型，forward 返回含学生 scalar loss 的 dict。"""
    return VJEPA(MiniVideoViT(embed_dim=32,depth=1,num_heads=4),LatentPredictor(32,predictor_dim=24,depth=1,num_heads=4))
