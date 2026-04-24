import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self,dim:int ,mlp_ratio:float=4.0,dropout:float=0.0 ) :

        super().__init__()
        dim_h=int(dim*mlp_ratio)
        self.fn1=nn.Linear(dim,dim_h)
        self.act=nn.GELU()
        self.fn2=nn.Linear(dim_h,dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        #x (b,n,d)
        x=self.fn1(x)
        x=self.act(x)
        x=self.dropout(x)
        x=self.fn2(x)
        x=self.dropout(x)
        return x
    

class MHA(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        assert dim%num_heads==0

        self.dim=dim
        self.num_heads=num_heads
        self.dhead=dim//num_heads
        self.scale=dim**-0.5

        self.qkv=nn.Linear(dim,3*dim)
        self.proj=nn.Linear(dim,dim)

        self.attndrop=nn.Dropout(dropout)
        self.projdrop=nn.Dropout(dropout)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        #x (b,n,dim)
        b,n,d=x.shape

        qkv=self.qkv(x)
        qkv=qkv.reshape(b,n,3,self.num_heads,self.dhead)
        qkv=qkv.permute(2,0,3,1,4)
        #(3,b,num_heads,n,dhead)
        q,k,v=qkv[0],qkv[1],qkv[2]

        #(b,num_heads,seq_len,dhead)
        attn=(q@k.transpose(-2,-1))*self.scale
        #->(b,n_heads,seq_len,seq_len)
        score=attn.softmax(dim=-1)
        score=self.attndrop(score)
        #->(b,n_heads,seq_len,dhead)
        out=score@v
        out=out.transpose(1,2).reshape(b,n,d)

        out=self.proj(out)
        out=self.projdrop(out)

        return out
    
class TransformerEncoderblock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1=nn.LayerNorm(dim)
        self.attn=MHA(dim,num_heads,dropout=dropout)

        self.norm2=nn.LayerNorm(dim)
        self.ffn=FFN(dim,mlp_ratio=mlp_ratio,dropout=dropout)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=x+self.attn(self.norm1(x))
        x=x+self.mlp(self.norm2(x))
        return x
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size=img_size
        self.patch_size=patch_size
        self.grid_size=img_size//patch_size

        self.num_patches=self.grid_size*self.grid_size

        self.proj=nn.Conv2d(in_chans,
                            embed_dim,
                            kernel_size=patch_size,
                            stride=patch_size)
        
    def forward(self,x: torch.Tensor)-> torch.Tensor:
        #x (b,3,224,224)
        x=self.proj(x)

        #x (b,786,14,14)
        x=x.flatten(2)
        #->(b,768,196)
        x=x.transpose(1,2)
        return x

class VIT(nn.Module):
    def __init__(self,
                 img_size:int=224,
                 patch_size:int=16,
                 in_chans:int=3,
                 num_cls:int=1000,
                 emb_dim:int=768,
                 depth:int=12,
                 num_heads:int=12,
                 mlp_ratio:float=4.0,
                 dropout:float=0.0):
        super().__init__()

        self.patch_emb=PatchEmbed(img_size=img_size,
                                  patch_size=patch_size,
                                  in_chans=in_chans,
                                  embed_dim=emb_dim)
        num_patches=self.patch_emb.num_patches
        self.cls_token=nn.Parameter(torch.zeros(1,1,emb_dim))

        #position emb
        self.pos_emb=nn.Parameter(torch.zeros(1,num_patches+1,emb_dim))

        self.pos_drop=nn.Dropout(dropout)

        self.blocks=nn.ModuleList(
            [
                TransformerEncoderblock(
                    dim=emb_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
                for _ in range(depth)
            ])
        
        self.norm=nn.LayerNorm(emb_dim)
        self.head=nn.Linear(emb_dim,num_cls)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        #x (b,3,224,224)
        b=x.shape[0]

        #patch
        x=self.patch_emb(x)
        #->(b,196,768)
        #196个position
        cls_token=self.cls_token.expand(b,-1,-1)
        #(1,1,768)->(b,1,768)
        x=torch.cat((cls_token,x),dim=1)
        #(b,197,768)

        x=x+self.pos_emb
        x=self.pos_drop(x)

        for block in self.blocks:
            x=block(x)
            #still (b,197,768)

        x=self.norm(x)
        #取所有样本的第0个token
        #因为使用cat[cls_token,x]
        #所以cls是第0token
        cls=x[:,0]
        #(b,768)
        out=self.head(cls)
        #(b,num_cls)
        return out
    