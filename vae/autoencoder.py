import torch
from torch import nn
from torch.nn import functional as F

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_c,out_c):
        super().__init__()

        # norm1,conv1
        self.norm1=nn.GroupNorm(32,in_c )
        self.conv1=nn.Conv2d(in_c,out_c,kernel_size=3,padding=1)

        # actva
        self.act=nn.SiLU()
        # norm2,conv2
        self.norm2=nn.GroupNorm(32,out_c)
        self.conv2=nn.Conv2d(out_c,out_c,kernel_size=3,padding=1)

        #solve skip dim diff

        if in_c!=out_c:
            self.skip=nn.Conv2d(in_c,out_c,kernel_size=1,padding=0)
        else:
            self.skip=nn.Identity()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        residual=x
        x=self.conv1(self.act(self.norm1(x)))
        x=self.conv2(self.act(self.norm2(x)))

        residual=self.skip(residual)
        return x+residual
    
class Residualblock(nn.Module):
    def __init__(self, inc=3,outc=3):
        super().__init__()
        self.norm1=nn.GroupNorm(32,inc)
        self.norm2=nn.GroupNorm(32,outc)

        self.conv1=nn.Conv2d(inc,outc,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(outc,outc,kernel_size=3,padding=1)

        if inc!=outc:
            self.residual=nn.Conv2d(inc,outc,kernel_size=1,padding=0)
        else:
            self.residual=nn.Identity()
    
    def forward(self,x):
        residual=x
        x=self.conv1(self.norm1(x))
        x=self.conv2(self.norm2(x))
        resiudal=self.residual(residual)

        return x+residual
    
class MHSA(nn.Module):
    def __init__(self, dmodel=768,nheads=1):
        super().__init__()

        self.dmodel=dmodel
        self.nheads=nheads
        
        assert dmodel%nheads==0 ,"dmodel//nheads must be integ"
        self.ndim=dmodel//nheads
        self.scale=self.ndim**-0.5
        self.inlinear=nn.Linear(dmodel,3*dmodel)
        self.out=nn.Linear(dmodel,dmodel)

    def forward(self,x):
        #shape 
        #(b,seq_len,dmodel)
        initial_shape=x.shape
        b,seq_len,dmodel=x.shape
        split_shape=(b,seq_len,self.nheads,self.ndim)
        q,k,v=self.inlinear(x).chunk(3,dim=-1)
        #(b,seq,dheads)
        q=q.view(split_shape).transpose(1,2)
        k=k.view(split_shape).transpose(1,2)
        v=v.view(split_shape).transpose(1,2)

        weight=q@k.transpose(-1,-2)
        score=weight*self.scale
        score=F.softmax(score,dim=-1)
        attn=score@v

        out=attn.transpose(1,2)
        out=out.contiguous().reshape(initial_shape)

        out=self.out(out)

        return out
    
class AEencoder(nn.Module):
    def __init__(self,inc=1,factor=8,latent=64):
        super().__init__()
        self.encode=nn.ModuleList(
        nn.Conv2d(inc,128,kernel_size=3,padding=1),

        #still
        Residualblock(128,128),
        Residualblock(128,128),

        #compress
        nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),

        #improve channel
        Residualblock(128,256),
        Residualblock(256,256),

        #compress
        nn.Conv2d(256,kernel_size=3,stride=2,padding=0),

        #
        Residualblock(256,512),
        Residualblock(512,512),

        #compress (1/8)
        nn.Conv2d(512,kernel_size=3,stride=2,padding=0),
        Residualblock(512,512),
        Residualblock(512,512),
        Residualblock(512,512),

        MHSA(dmodel=512,nheads=2),
        Residualblock(512,512),
        nn.GroupNorm(32,512),

        nn.SiLU(),
        nn.Conv2d(512,latent,kernel_size=3,padding=1)
        

        )
    def forward(self,x):
        x=self.encode(x)
        return x
    
class AEdecoder(nn.Module):
