import math
import torch
import torch.nn as nn
import torch.nn.functional as F




"""
input x_t : (b,c,h,w)
t:        : (b)
unet(x_t,t)->noise :(b,c,h,w)
"""


def timestep_emb(timestep:torch.Tensor,dim:int)->torch.Tensor:
    """
    t->t_emb
    t : (b)
    t_emb: (b,dim)

    """
    #use cosine,sin position emb
    half=dim//2
    device=timestep.device

    #sin(pos/freqs)
    #where freqs=10000**(-2*i/dmodel)


    freqs=torch.exp(
        -math.log(10000)*torch.arange(half,device).float()/half
    )

    #pos (postion)
    #compute the pos/freqs
    args=timestep.float().unsqueeze(1)*freqs.unsqueeze(0)
    #timestep 1 d tensor use unsqueeze(1)
    #(b)-->(b,1)
    #so,freqs 1d 
    #(const)-->(0,const)
    emb=torch.cat([torch.sin(args),torch.cos(args)],dim=1)

    if dim%2==0:
        emb=F.pad(emb,(0,1))
    return emb


class Silu(nn.Module):
    def forward(self,x):
        return x*torch.sigmoid(x)


"""
in_c,
out_c,
emb_c,

"""

class Resblock(nn.Module):
    def __init__(self, in_c:int,out_c:int,emb_c:int):
        super().__init__()

        self.in_c=in_c
        self.out_c=out_c
        self.emb_c=emb_c

        self.norm1=nn.GroupNorm(32 if in_c>=32 else 1,in_c)
        self.act1=Silu()
        self.conv1=nn.Conv2d(in_c,out_c,3,1)

        self.emb_proj=nn.Sequential(
            Silu(),
            nn.linear(emb_c,out_c)
        )

        self.norm2=nn.GroupNorm(32 if in_c>=32 else 1,in_c)
        self.act2=Silu()
        self.conv2=nn.Conv2d(out_c,out_c,3,1)

        if in_c!=out_c:
            #solve the shortcut dim diff
            self.skip=nn.Conv2d(in_c,out_c,kernel_size=1)
        else:
            self.skip=nn.Identity()
        
    def forward(self,x:torch.Tensor,emb:torch.Tensor):
        h=self.conv1(self.act1(self.norm1(x)))

        emb_out=self.emb_proj(emb)[:,:,None,None]
        #(b,out_c,1,1)
        h=h+emb_out
        h=self.conv2(self.act2(self.norm2(h)))

        return h+self.skip(x)
    
class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv=nn.Conv2d(ch,ch,kernel_size=3,stride=2,padding=1)
    
    def forward(self,x):
        return self.conv(x)
    

class Upsample(nn.Module):
    def __init__(self, ch:int):
        super().__init__()
        self.conv=nn.Conv2d(ch,ch,kernel_size=3,padding=1)
    
    def forward(self,x):
        x=F.interpolate(x,scale_factor=2,mode="nearest")


        return self.conv(x)


