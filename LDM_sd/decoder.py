import torch
from torch import nn
from torch.nn import functional as F
from attention import MHSA

"""
VAE_AttentionBlock
    input -(b,channel,h,w)
        vae-attn(channel)
    output still


decoder

    input   -(b,4,h/8,w/8)
    output -(b,3,h,w)

    (b,4,h/8,w/8)-->(b,512,h/8,w/8)
    attn,res
    upsample1->(b,512,h/4,w/4)
    res(512,512)
    upsample2->(b,512,h/2,w/2)
    res(512,256)
    upsample ->(b,256,h,w)
    res(256,128)

vae_residual

    res(in_c,out_c),with residual


"""
class VAE_AttentionBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.norm=nn.GroupNorm(32,channel)
        self.attn=MHSA(1,dim=channel)

    def forward(self,x):
        residaul=x
        x=self.norm(x)
        #attn 
        #(b,seq_len,dim)
        b,c,h,w=x.shape
        #(b,c,h,w)-->(b,c,h*w)
        x=x.view((b,c,h*w))
        x=x.transpose(-1,-2)
        x=self.attn(x)
        #(b,h*w,c)-->(b,c,h*w)
        x=x.transpose(-1,-2)
        #(b,c,h*w)-->(b,c,h,w)
        x=x.view((b,c,h,w))
        x+=residaul

        return x




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
    


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            #initial proces
            nn.Conv2d(4,4,kernel_size=1,padding=0),

            nn.Conv2d(4,512,kernel_size=3,padding=1),

            #res+atttn+res*4
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            #upsample1
            #(b,512,h/8,w/8)-->(b,512,h/4,w/4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),

            #3 res
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),


            #upsample 2
            #(b,512,h/4,w/4)->(b,512,h/2,w/2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),

            #compress channel
            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),

            #upsample3
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,256,kernel_size=3,padding=1),

            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            nn.GroupNorm(32,128),
            nn.SiLU(),
            #back 3
            nn.Conv2d(128,3,kernel_size=3,padding=1)
        )
    def forward(self,x):
        #x (b,4,h/8,w/8)-->(b,3,h,w)
        x=x/0.18215
        for layer in self:
            x=layer(x)

        return x
























