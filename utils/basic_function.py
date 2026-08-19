import torch
import math
from torch import nn
from torch.nn import functional as F
try:
    from .attention import SpatialMHSA, SpatialTransformer
    from .nn import make_group_norm
except ImportError:
    # Supports `python utils/basic_function.py` for local shape tests.
    from attention import SpatialMHSA, SpatialTransformer
    from nn import make_group_norm





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
            h=self.norm2(h)
            h=h+time_emb[:,:,None,None]
            h=F.silu(h)

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

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10_000):
        super().__init__()
        self.dim=dim
        self.max_period=max_period

    def forward(self,timesteps):
        half_dim=self.dim//2

        frequencies=torch.exp(
            -math.log(self.max_period)
            *torch.arange(half_dim,device=timesteps.device)/half_dim
        )

        angles=timesteps.float()[:,None]*frequencies[None,:]

        embedding=torch.cat(
            [torch.cos(angles),torch.sin(angles)],
            dim=-1
        )

        if self.dim%2==1:
            embedding=F.pad(embedding,(0,1))
        
        return embedding
    
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv=nn.Conv2d(channels,channels,kernel_size=3,stride=2,padding=1)

    def forward(self,x):
        return self.conv(x)
    
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv=nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x):
        x=F.interpolate(x,scale_factor=2,mode="nearest")
        return self.conv(x)
    

class Downblock(nn.Module):
    def __init__(self,
                 in_channels,out_channels,
                  time_dim,
                   use_attn=False,
                    context_dim=None, ):
        super().__init__()

        self.res1=TimeResidualblock(in_channels,out_channels,time_dim)
        self.res2=TimeResidualblock(out_channels,out_channels,time_dim)

        self.attn=(SpatialTransformer(
            channels=out_channels,
            num_heads=8,
            head_dim=out_channels//8,
            depth=1,
            context_dim=context_dim,
        ) 
        if use_attn
        else None

        )
        self.downsample=Downsample(out_channels)

    def forward(self,x,time_emb,context=None):

        x=self.res1(x,time_emb)
        x=self.res2(x,time_emb)

        if self.attn is not None:
            x=self.attn(x,context=context)

        skip=x
        x=self.downsample(x)
        return x,skip

class Upblock(nn.Module):
    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                  time_dim,
                   use_attn=False,
                    context_dim=None, ):
        super().__init__()

        self.upsample=Upsample(in_channels)
        self.res1=TimeResidualblock(in_channels+skip_channels,out_channels,time_dim)
        self.res2=TimeResidualblock(out_channels,out_channels,time_dim)

        self.attn=(
            (SpatialTransformer(
                channels=out_channels,
                num_heads=8,
                head_dim=out_channels//8,
                depth=1,
                context_dim=context_dim,
            ))
            if use_attn
            else None
        )

    def forward(self,x,skip,time_emb,context=None):
        x=self.upsample(x)

        x=torch.cat([x,skip],dim=1)

        x=self.res1(x,time_emb)
        x=self.res2(x,time_emb)

        if self.attn is not None:
            x=self.attn(x,context=context)
        return x

class TimeUNet(nn.Module):
    """
    适用于 noise / score / velocity prediction。

    x_t:     (B, C, H, W)
    t:       (B,)
    context: (B, L_context, context_dim)，无条件训练时为 None
    output:  (B, C, H, W)
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_dim=256,
        context_dim=768,
    ):
        super().__init__()

        self.time_embedding=nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels,time_dim),
            nn.SiLU(),
            nn.Linear(time_dim,time_dim),
        )

        self.input_conv=nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.down1=Downblock(in_channels=base_channels,
                             out_channels=base_channels,
                             time_dim=time_dim,
                             use_attn=False,
                             context_dim=context_dim,)
        self.down2=Downblock(in_channels=base_channels,
                             out_channels=base_channels*2,
                             time_dim=time_dim,
                             use_attn=False,
                             context_dim=context_dim,
                             )
        
        self.down3=Downblock(in_channels=base_channels*2,
                             out_channels=base_channels*4,
                             time_dim=time_dim,
                             use_attn=True,
                             context_dim=context_dim,)
        
        self.mid1=TimeResidualblock(
            base_channels*4,
            base_channels*4,
            time_dim=time_dim,
        )
        self.midattn=SpatialTransformer(
            channels=base_channels*4,
            num_heads=8,
            head_dim=(base_channels*4)//8,
            depth=1,
            context_dim=context_dim,

        )
        self.mid2=TimeResidualblock(
            base_channels*4,
            base_channels*4,
            time_dim,
        )

        self.up3 = Upblock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
            time_dim=time_dim,
            use_attn=True,
            context_dim=context_dim,
        )

        # 16×16 -> 32×32，无 attention
        self.up2 = Upblock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
            time_dim=time_dim,
            use_attn=False,
            context_dim=context_dim,
        )

        # 32×32 -> 64×64，无 attention
        self.up1 = Upblock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
            time_dim=time_dim,
            use_attn=False,
            context_dim=context_dim,
        )

        self.out=nn.Sequential(
            make_group_norm(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels,out_channels,kernel_size=3,padding=1)

        )

    def forward(self,x_t,timesteps,context=None):
        time_emb=self.time_embedding(timesteps)

        x=self.input_conv(x_t)
        x,skip1=self.down1(x,time_emb,context)
        x,skip2=self.down2(x,time_emb,context)
        x,skip3=self.down3(x,time_emb,context)

        x=self.mid1(x,time_emb)
        x=self.midattn(x,context=context)
        x=self.mid2(x,time_emb)

        x=self.up3(x,skip3,time_emb,context)
        x=self.up2(x,skip2,time_emb,context)
        x=self.up1(x,skip1,time_emb,context)

        return self.out(x)

"""
Old inline test kept as a comment. Tests must not run while importing utils.
model = TimeUNet(
    in_channels=3,
    out_channels=3,
    base_channels=64,
    time_dim=256,
    context_dim=768,
)

x_t = torch.randn(2, 3, 64, 64)
timesteps = torch.randint(0, 1000, (2,))

# 当前无条件训练
prediction = model(x_t, timesteps, context=None)

print(prediction.shape)
# torch.Size([2, 3, 64, 64])
"""

if __name__ == "__main__":
    model = TimeUNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_dim=256,
        context_dim=768,
    )
    x_t = torch.randn(2, 3, 64, 64)
    timesteps = torch.randint(0, 1000, (2,))
    prediction = model(x_t, timesteps, context=None)
    print(prediction.shape)
