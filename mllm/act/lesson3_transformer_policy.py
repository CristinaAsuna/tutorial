"""关卡 3：观测 memory 和并行 action queries。
目标：多相机位置不混淆，用 K 个 query 一次预测 K 步。
前置：reshape/flatten、Conv patch tokens；encoder/decoder 直接复用 PyTorch。
形状：images(B,Cam,3,H,W) -> patches(B,Cam*N,D)；qpos/z 各一个 token；memory(B,Cam*N+2,D)。
手算：B=2,Cam=2,N=4 -> 视觉8个+qpos1个+z1个=10个 memory token；K=5输出(B,5,A)。
常见错：漏 camera embedding；把 action query 当做示范动作；给并行 decoder 添加 next-token shift。
接入：practice Policy 覆盖 _encode_observation、_decode，调用本文件函数。
"""
import torch

def encode_observation(policy, images, qpos, z):
    # policy 已提供 visual,spatial_pos,camera_embed,qpos_proj,z_proj,memory_encoder。
    # TODO 3.1：验证 (B,Cam,3,H,W)、分辨率、qpos shape 和 camera 上限。
    # TODO 3.2：flatten B/Cam，visual -> flatten空间/transpose 得 (B*Cam,N,D)，恢复相机轴。
    # TODO 3.3：加空间位置和 camera embedding，再变 (B,Cam*N,D)。
    # TODO 3.4：拼 qpos/z 投影 token，送 memory_encoder。
    raise NotImplementedError("ACT TODO 3.1-3.4: encode_observation")

def decode_action_chunk(observation_memory, action_queries, decoder, action_head):
    # TODO 3.5：queries(1,K,D) expand 到 B；decoder(queries,memory) -> action_head -> (B,K,A)。
    raise NotImplementedError("ACT TODO 3.5: decode_action_chunk")
