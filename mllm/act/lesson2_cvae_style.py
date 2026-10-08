"""关卡 2：CVAE posterior、采样和目标函数。
目标：训练时从 demonstration 推断 z；推理时完全不用 demonstration。
前置：Linear、cat、Transformer padding mask、高斯参数；模块初始化已经给出。
形状：qpos(B,Q), actions(B,K,A), valid(B,K) -> mean/logvar(B,Z) -> z(B,Z)。
手算：mean=0,logvar=0 时 KL=0；mean=1,logvar=0,Z=1 时 KL=0.5；noise=0 时 z=mean。
常见错：logvar 是方差的对数；torch encoder 的 padding mask True 表示忽略，和 edu_core 相反。
接入：practice Policy 使用本关 StyleEncoder.forward、reparameterize、cvae_loss。
"""
import torch
from torch import nn
import torch.nn.functional as F

class StyleEncoder(nn.Module):
    def __init__(self, qpos_dim, action_dim, chunk_size, dim, latent_dim, heads):
        super().__init__()
        self.qpos_proj=nn.Linear(qpos_dim,dim)
        self.action_proj=nn.Linear(action_dim,dim)
        self.cls=nn.Parameter(torch.zeros(1,1,dim))
        self.pos=nn.Parameter(torch.zeros(1,chunk_size+2,dim))
        layer=nn.TransformerEncoderLayer(dim,heads,dim*2,batch_first=True,dropout=0.,activation="gelu")
        self.encoder=nn.TransformerEncoder(layer,1)
        self.mean=nn.Linear(dim,latent_dim)
        self.logvar=nn.Linear(dim,latent_dim)
        nn.init.normal_(self.cls,std=.02)
        nn.init.normal_(self.pos,std=.02)

    def forward(self, qpos, actions, action_mask):
        # TODO 2.1：验证 shape；拼 [CLS,qpos,action_0,...]；CLS expand 到 B。
        # TODO 2.2：前两个位置恒有效；拼 action_mask；加 pos，encoder(...,src_key_padding_mask=~valid)。
        # TODO 2.3：仅取 CLS 输出，分别送 mean/logvar head。
        raise NotImplementedError("ACT TODO 2.1-2.3: StyleEncoder.forward")

def reparameterize(mean, logvar, noise=None):
    # TODO 2.4：noise 默认 randn_like(mean)；z=mean+exp(0.5*logvar)*noise。
    raise NotImplementedError("ACT TODO 2.4: reparameterize")

def cvae_loss(predicted_actions, target_actions, action_mask, mean, logvar, beta):
    # TODO 2.5：扩展 valid 到动作维度，只对有效标量计算 L1；无有效 target 抛 ValueError。
    # TODO 2.6：KL=0.5*(mean^2+exp(logvar)-1-logvar)，Z 求和、B 求均值。
    # TODO 2.7：返回 (L1+beta*KL,L1,KL)，便于检查两个损失而非只看总和。
    raise NotImplementedError("ACT TODO 2.5-2.7: cvae_loss")
