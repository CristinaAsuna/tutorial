"""关卡 1：把一条动作轨迹切成每个时刻的未来 action chunk。"""
import torch


def sample_future_chunks(actions: torch.Tensor, chunk_size: int):
    # TODO: 返回 (B,T,K,A) chunk 及 (B,T,K) valid mask；末尾不可越界。
    raise NotImplementedError("参考 reference_act.sample_action_chunks")
