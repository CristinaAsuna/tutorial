import torch
from torch import nn
from torch.nn import functional as F
from utils.attention import SpatialMHSA,SpatialTransformer


def make_group_norm(channels,max_groups):
    """
    make sure channels%group=0
    """
    groups=min(channels,max_groups)
    while channels%groups !=0:
        groups-=1
    return nn.GroupNorm(groups,channels,eps=1e-6)





class Residualblock(nn.Module):
    def __init__(self, inc, outc, dropout=0.0):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, inc, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, outc, eps=1e-6)

        self.conv1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv2d(inc, outc, kernel_size=1)
            if inc != outc else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)

        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))

        return x + residual
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=256, code_dim=32, beta=0.25):
        super().__init__()
        """
    return 
        z_q
        indices
        vq_loss
    """
    #parameterize codebook,(k,d) tensor
        self.num_codes=num_codes
        self.code_dim=code_dim
        self.embedding=nn.Embedding(num_codes,code_dim)
        nn.init.uniform_(self.embedding.weight,-1/num_codes,1/num_codes)
        self.beta=beta

    def forward(self,z_e):
        #z_e (b,c,h,w)-->(b,h,w,c)-->(b*w*h,c)
        b,c,h,w=z_e.shape
        if c != self.code_dim:
            raise ValueError(f"code_dim 应为 {self.code_dim}，实际为 {c}")
        z_flat=z_e.permute(0,2,3,1).contiguous().view(-1,c)

        codebook=self.embedding.weight

        #compute distance
        #(b*h*w,k)
        distance=(z_flat.pow(2).sum(dim=1,keepdim=True)
                  +codebook.pow(2).sum(dim=1)
                  -2*z_flat@codebook.t())
        
        # find index
        #(b*h*w,)
        flat_indices=distance.argmin(dim=1)
        # search by index,return zq
        #(b*h*w,dim)
        #z_flat[i] 被 codebook[flat_indices[i]] 替换
        z_q=self.embedding(flat_indices)

        #reshape z_q--->(b,dim,h,w),where dim=c,-->(b,c,h,w)
        z_q=z_q.view(b,h,w,c).permute(0,3,1,2).contiguous()
        # permute 就是当前位置放前面的index

        #loss
        #但认为encoder,embedding学习速率不一样,将其拆分
        #ze <---> zq, close
        # codebook loss 只更新 embedding；commitment loss 只更新 encoder。
        codebook_loss=F.mse_loss(z_q,z_e.detach())
        commitment_loss=F.mse_loss(z_e,z_q.detach())
        vq_loss=codebook_loss + self.beta*commitment_loss

        indices=flat_indices.view(b,h,w)

        counts=torch.bincount(flat_indices,minlength=self.num_codes).float()
        probs=counts/counts.sum().clamp_min(1.0)
        perplexity=torch.exp(-(probs*torch.log(probs+1e-10)).sum())
        metrics={
            "vq":vq_loss.detach(),
            "codebook":codebook_loss.detach(),
            "commitment":commitment_loss.detach(),
            "perplexity":perplexity.detach(),
            "active_codes":(counts>0).sum().detach(),
        }

        return z_q,indices,vq_loss,metrics
    
class TimeResidualblock(nn.Module):
    """
    x:        (B, in_channels, H, W)
    time_emb: (B, time_dim)
    output:   (B, out_channels, H, W)
    """
    def __init__(self, inc, outc, 
                 time_dim,
                 dropout=0.0,
                 use_scale_shift=True):
        super().__init__()
        self.use_scale_shift=use_scale_shift
        """
        Normalize(feature)
        → feature * (1 + condition_scale)
        → feature + condition_shift
        """
        self.norm1=make_group_norm(inc)
        self.norm2=make_group_norm(outc)
        self.conv1=nn.Conv2d(inc,outc,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(outc,outc,kernel_size=3,padding=1)

        self.dropout=nn.Dropout(dropout)

        #scale shift,need 2*out channel
        time_out_dim= 2*outc if use_scale_shift else outc

        if inc!=outc:
            self.skip=nn.Conv2d(inc,outc,kernel_size=1)
        else:
            self.skip=nn.Identity()

        self.time_proj=nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim,time_out_dim)
        )

    def forward(self,x,time):
        residual=self.skip(x)

        h=self.conv1(F.silu(self.norm1(x)))

        #time proj
        time_emb=self.time_proj(time).to(dtype=h.dtype)

        if self.use_scale_shift:
            scale,shift=time_emb.chunk(2,dim=1)
            #(b,c)-->(b,c,1,1)
            # because of broadcasting, (b,c,1,1)-->(b,c,h,w)
            #[:, :, None, None] 是在末尾插入两个长度为 1 的维度：
            h=self.norm2(h)
            h=h*(1+scale[:,:,None,None])
            h=h+shift[:,:,None,None]
            h=F.silu(h)
        else:
            h=h+time_emb[:,:,None,None]
            h=F.silu(self.norm2(h))

        h=self.conv2(self.dropout(h))

        return h+residual

class SwitchSequantial(nn.Sequential):
    def forward(self,x,time,text):
        for layer in self:
            if isinstance(layer,TimeResidualblock):
                layer=layer(x,time)
            if isinstance(layer,SpatialTransformer):
                layer=layer(x,text)
            else:
                layer=layer(x)

