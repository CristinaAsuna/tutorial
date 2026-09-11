import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

"""
encoder:
    input(x,noise-forward)
    -(b,3,h,w)
    output(feature x)
    -(b,4,h/f,w/f),where f==8
"""

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(


            #(b,3,h,w)->(b,128,h,w)
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            #2 of res,still shape
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            #step2
            #compress->(b,128,h/2,w/2)
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
            
            #two res
            #1.(128->256)
            #2.(256,256)
            VAE_ResidualBlock(128,256),
            VAE_ResidualBlock(256,256),

            #replicate step2
            #256->512  ->(b,512,h/4,w/4)
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),

            VAE_ResidualBlock(256,512),
            VAE_ResidualBlock(512,512),

            #compress,3*res_still
            #->(b,512,h/8,w/8)
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            #attn
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            nn.GroupNorm(32,512),

            nn.SiLU(),

            #copmress channel->8
            nn.Conv2d(512,8,kernel_size=3,padding=1),

            #use 1x1
            nn.Conv2d(8,8,kernel_size=1,padding=0),


        )
    
    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if getattr(layer,'stride',None)==(2,2):
                #solve padding un
                x=F.pad(x,(0,1,0,1))
            x=layer(x)
        return x

    def encode_stats(self, x: torch.Tensor, noise: torch.Tensor):
        x = self._encode_features(x)

        #compulate mean,var
        #chunk from channel
        #maen,var (b,4,h/8,w/8)
        mean,log_var=torch.chunk(x,2,dim=1)
        log_var=torch.clamp(log_var,-30,20)
        var=log_var.exp()

        std=var.sqrt()
        #noise ~(0,1)
        #z     ~(mean,std**2)
        #to get z,compulate z=mean+std*nosie
        latents=mean+std*noise

        latents*=0.18215

        return latents, mean, log_var

    def forward(self,x:torch.Tensor,noise:torch.Tensor)->torch.Tensor:
        
        #x (b,3,h,w)-->(b,8,h/8,w/8)
        x, _, _ = self.encode_stats(x, noise)

        #x (b,4,h/8,w/8)
        return x
    



