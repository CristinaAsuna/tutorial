import torch
from torch import nn
from torch.nn import functional as F
from utils.basic_function import make_group_norm
class SpatialMHSA(nn.Module):
    def __init__(self, channels, nheads=8, dropout=0.0):
        super().__init__()
        assert channels % nheads == 0

        self.nheads = nheads
        self.head_dim = channels // nheads
        self.dropout = dropout

        self.norm = nn.GroupNorm(32, channels)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x):
        # x: (B, C, H, W)
        residual = x
        b, c, h, w = x.shape

        # 先归一化，再把空间位置展平为 token
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = q.view(b, h * w, self.nheads, self.head_dim).transpose(1, 2)
        k = k.view(b, h * w, self.nheads, self.head_dim).transpose(1, 2)
        v = v.view(b, h * w, self.nheads, self.head_dim).transpose(1, 2)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        x = x.transpose(1, 2).contiguous().view(b, h * w, c)
        x = self.to_out(x)

        x = x.transpose(1, 2).reshape(b, c, h, w)

        # 注意力模块通常应保留残差连接
        return x + residual
    
def zero_module(module):
    """
    initial residual
    """
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class GEGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.proj=nn.Linear(dim,hidden_dim*2)
        self.out=nn.Linear(hidden_dim,dim)

    def forward(self,x):
        x,gate=self.proj(x).chunk(2,dim=-1)
        return self.out(x*F.gelu(gate))

class MultiheadAttention(nn.Module):
    """
    x : b,n_q,dim
    context: b, seq_len, context_dim
    output: b, n_q,dim

    when context==none,-->selfattn
    """
    def __init__(self, 
                 dim,num_heads=8,
                 head_dim=64,
                 context_dim=None,
                 dropout=0.0):
        super().__init__()
        innerdim=num_heads*head_dim
        self.inner_dim=num_heads*head_dim
        self.num_heads=num_heads
        self.head_dim=head_dim

        self.dropout=dropout
        #q,  
        # k,v for context
        self.q=nn.Linear(dim,innerdim)
        context_dim=dim if context_dim is None else context_dim

        self.k=nn.Linear(context_dim,innerdim)
        self.v=nn.Linear(context_dim,innerdim)

        self.out=nn.Sequential(
            nn.Linear(innerdim,dim),
            nn.Dropout(dropout),
        )

    def forward(self,x,context=None):

        if context is None:
            context=x
        
        b,seq_len,_=x.shape
        context_len=context.shape[1]

        q=self.q(x)
        #
        k=self.k(context)
        v=self.v(context)

        #seq_len ?= context_len
        # (B, N, H*D) -> (B, H, N, D)
        q=q.view(b,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        
        k=k.view(b,context_len,self.num_heads,self.head_dim).transpose(1,2)
        v=v.view(b,context_len,self.num_heads,self.head_dim).transpose(1,2)

        x=F.scaled_dot_product_attention(q,
                                         k,
                                         v,
                                         dropout_p=self.dropout if self.training else 0.0)
        
        x=x.transpose(1,2).contiguous()
        x=x.view(b,seq_len,-1)

        return self.out(x)
    

class Transformerblock(nn.Module):
    """
    vanisal decoder-only
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=64,
        context_dim=None,
        dropout=0.0,
    ):
        super().__init__()

        self.norm1=nn.LayerNorm(dim)
        self.self_attn=MultiheadAttention(dim=dim,
                                          num_heads=num_heads,head_dim=head_dim,
                                          context_dim=None,dropout=dropout)
        
        self.norm2=nn.LayerNorm(dim)
        self.cross_attn=MultiheadAttention(dim=dim,
                                          num_heads=num_heads,head_dim=head_dim,
                                          context_dim=context_dim,dropout=dropout)
        
        self.norm3=nn.LayerNorm(dim)
        self.ffn=GEGLU(dim,hidden_dim=dim*4)

    def forward(self,x,context=None):
        residual=x
        x=residual+self.self_attn(self.norm1(x))

        #cross or self
        if context is not None:
            x=x+self.cross_attn(self.norm2(x),context=context)
        
        x=x+self.ffn(self.norm3(x))

        return x
    

class SpatialTransformer(nn.Module):
    """
    x:       (B, C, H, W)
    context: (B, L_context, context_dim)，可为 None
    output:  (B, C, H, W)
    """

    def __init__(
        self,
        channels,
        num_heads=8,
        head_dim=32,
        depth=1,
        context_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim=num_heads*head_dim

        self.norm=make_group_norm(channels)
        self.proj_in=nn.Conv2d(channels,inner_dim,kernel_size=1
                               )
        
        self.blocks=nn.ModuleList(
            [
                Transformerblock(dim=inner_dim,
                                 num_heads=num_heads,
                                 head_dim=head_dim,
                                 context_dim=context_dim,
                                 dropout=dropout)
                                 for _ in range(depth)
                                
            ]
        )

        self.proj_out=zero_module(
            nn.Conv2d(inner_dim,channels,kernel_size=1)
        )
    
    def forward(self,x,context=None):
        residual=x
        b,_,h,w=x.shape

        x=self.proj_in(self.norm(x))

        # (B, C, H, W) -> (B, C, H*W,)->(B, H*W, C)
        x=x.flatten(2).transpose(1,2)


        for block in self.blocks:
            x=block(x,context=context)

        x=x.transpose(1,2).reshape(
            b,-1,h,w
        )

        return self.proj_out(x)+residual