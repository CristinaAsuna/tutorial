"""关卡 1：未来动作窗口。
目标：每个观测时刻得到未来 K 个 absolute joint targets。
前置：arange、广播、高级索引；不需要重写 attention。
形状：actions (B,T,A) -> chunks (B,T,K,A), valid (B,T,K) bool。
手算：actions=[10,20,30], K=2 -> [[10,20],[20,30],[30,30]]；最后 mask=[1,0]。
常见错：把无效末尾动作也计入 loss；把 chunk 当作相对位移。
接入：practice_act.sample_action_chunks 是 demo 真正使用的数据准备入口。
"""
import torch

def sample_future_chunks(actions: torch.Tensor, chunk_size: int):
    # TODO 1.1：检查三维、T>0、K>0，创建 start(T,1) 与 offset(1,K)。
    # TODO 1.2：index=start+offset；valid=index<T；clamp 越界索引后取 actions。
    # TODO 1.3：valid 扩展到 B；返回 chunks、bool valid（device 与输入一致）。
    raise NotImplementedError("ACT TODO 1.1-1.3: sample_future_chunks")
