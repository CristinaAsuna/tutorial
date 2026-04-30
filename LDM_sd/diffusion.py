import torch
from torch import nn
from torch.nn import functional as F
from attention import MHSA, CrossAttention

"""
Diffusion(unet)
    time_emb
    unet
        -residual
        -unet attn(mhsa,crossattn)
        -downsample,bottleneck,upsample
    output





"""
class TimeEmbedding(nn.Module):
    def __init__(self ,dim) :
        super().__init__()
        #use linear twice
        self.ln1=nn.Linear(dim,4*dim)
        self.ln2=nn.Linear(dim*4,4*dim)


    def forward(self,time):
        #x (1,320)->(1,1280)
        x=self.ln1(x)
        x=F.silu(x)
        x=self.ln2(x)

        return x
    
class UNET_Outlayer(nn.Module):
    def __init__(self ,in_c,out_c) :
        super().__init__()
        self.conv=nn.Conv2d(in_c,out_c,kernel_size=3,padding=1)
        self.norm=nn.GroupNorm(32,in_c)

    def forward(self,x):
        #x (b,320,h/8,w/8)-->(b,4,h/8,w/8)
        x=self.norm(x)
        x=F.silu(x)
        x=self.conv(x)
        return x

"""
if ininstance (A,B)
    判断A是不是B的子类,
    等于说给了不同开关,控制一个layer接受哪一种参数()


"""
class Upsample(nn.Module):
    def __init__(self,channels) :
        super().__init__()
        self.conv=nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x):
        #x (b,c,h,w)->(b,c,h*2,w*2)
        x=F.interpolate(x,scale_factor=2,mode='nearest')
        return self.conv(x)
    
class UNET_Attn(nn.Module):
    def __init__(self,n_heads,n_emb,d_context) :
        super().__init__()
        channels=n_heads*n_emb

        self.groupnorm=nn.GroupNorm(32,channels,eps=1e-6)

        self.conv_input=nn.Conv2d(channels,channels,kernel_size=1,padding=0)

        self.ln1=nn.LayerNorm(channels)
        self.attn1=MHSA(n_heads,channels,in_proj_bias=False)
        self.ln2=nn.LayerNorm(channels)
        self.cross_attn=CrossAttention(n_heads,channels,d_context,in_proj_bias=False,out_proj_bias=False)

        self.ln3=nn.LayerNorm(channels)

        self.ffn1=nn.Linear(channels,channels*4*2)
        #2 plus 4
        self.ffn2=nn.Linear(4*channels,channels)

        self.conv_out=nn.Conv2d(channels,channels,kernel_size=1,padding=0)

    def forward(self,x,context):
        #x (b,c,h,w)
        # context (b,seq_len,dim)
        residual_long=x

        x=self.groupnorm(x)
        x=self.conv_input(x)
        b,c,h,w=x.shape
        x=x.view((b,c,h*w))
        x=x.transpose(-1,-2)

        #norm + selfattn + skip
        residual_short=x

        x=self.ln1(x)
        x=self.attn1(x)
        x+=residual_short

        residual_short=x

        # norm+ cross attn +skip
        x=self.ln2(x)
        x=self.cross_attn(x,context)
        x+=residual_short

        residual_short=x

        # norm + ffn with geglu +skip
        x=self.ln3(x)
        x,gate=self.ffn1(x).chunk(2,dim=-1)
        x=x*F.gelu(gate)

        #(b,h*w,c*4)->(b,h*w,c)
        x=self.ffn2(x)
        x+=residual_short
        #reshape
        x=x.transpose(-1,-2)
        x=x.view((b,c,h,w))

        return self.conv_out(x)+residual_long


class UNET_Residual(nn.Module):
    def __init__(self,in_c,out_c,time=1280) :
        super().__init__()
        self.norm1=nn.GroupNorm(32,in_c)
        self.merge_norm=nn.GroupNorm(32,out_c)

        self.conv1=nn.Conv2d(in_c,out_c,kernel_size=3,padding=1)
        self.merge_conv=nn.Conv2d(out_c,out_c,kernel_size=3,padding=1)

        self.linear_time=nn.Linear(time,out_c)

        if in_c!=out_c:
            self.skip=nn.Conv2d(in_c,out_c,kernel_size=1,padding=0)
        else :
            self.skip=nn.Identity()

    def forward(self,x,time):
        #x (b,c,h,w)
        #time (1,1280)
        residual =x
        x=self.norm1(x)
        x=F.silu(x)
        x=self.conv1(x)

        time=F.silu(time)
        merged=x+time.unsqueeze(-1).unsqueeze(-1)

        merged=self.merge_norm(merged) 
        merged=F.silu(merged)
        merged=self.merge_conv(merged) 

        return merged+self.skip(residual)


    


class SwitchSequential(nn.Sequential):
    def forward(self,x,context,time):
        for layer in self:
            if isinstance(layer,UNET_Attn):
                x=layer(x,context)
            elif isinstance(layer,UNET_Residual):
                x=layer(x,time)
            else:
                x=layer(x)
        return x

class UNET(nn.Module):
    def __init__(self) :
        super().__init__()
        self.encoder=nn.ModuleList([
        #(b,4,h/8,w/8)->(b,320,h/8,w/8)
        SwitchSequential(nn.Conv2d(4,320,kernel_size=3,padding=1)),
        #twice resnet,attn
        SwitchSequential(UNET_Residual(320,320),UNET_Attn(8,40)),
        SwitchSequential(UNET_Residual(320,320),UNET_Attn(8,40)),

        #compress h,w
        #(b,320,h/8,w/8)->(b,320,h/16,w/16)
        SwitchSequential(nn.Conv2d(320,320,kernel_size=3,stride=2,padding=1)),

        #res(320,640),attn(8,80)
        SwitchSequential(UNET_Residual(320,640),UNET_Attn(8,80)),
        SwitchSequential(UNET_Residual(640,640),UNET_Attn(8,80)),

        #compress h,w
        #(b,640,h/16,w/16)->(b,640,h/32,w/32)
        SwitchSequential(nn.Conv2d(640,640,kernel_size=3,stride=2,padding=1)),
        #res(640,1280),attn(8,160)
        SwitchSequential(UNET_Residual(640,1280),UNET_Attn(8,160)),
        SwitchSequential(UNET_Residual(1280,1280),UNET_Attn(8,160)),

        # compress h,w
        #(b,1280,h/32,w/32)->(b,1280,h/64,w/64)
        SwitchSequential(nn.Conv2d(1280,1280,kernel_size=3,stride=2,padding=1)),
        #twice res(1280,1280)
        SwitchSequential(UNET_Residual(1280,1280)),
        SwitchSequential(UNET_Residual(1280,1280)),


        ])

        self.bottleneck=SwitchSequential(
            #(b,1280,h/64,w/64)->(b,1280,h/64,w/64)
            #res,attn,res
            SwitchSequential(UNET_Residual(1280,1280)),
            SwitchSequential(UNET_Attn(8,160)),
            SwitchSequential(UNET_Residual(1280,1280)),
        )
        
        #kind of different 
        #cause of skip connection ,some inc while change to a+b
        #12 个 list,即12次skip connection
        """
        bottleneck 输出: 1280, H/64

decoder[0]  concat S11: 1280 + 1280 = 2560  -> 1280, H/64
decoder[1]  concat S10: 1280 + 1280 = 2560  -> 1280, H/64
decoder[2]  concat S9 : 1280 + 1280 = 2560  -> 1280, H/64 -> upsample 到 H/32

decoder[3]  concat S8 : 1280 + 1280 = 2560  -> 1280, H/32
decoder[4]  concat S7 : 1280 + 1280 = 2560  -> 1280, H/32
decoder[5]  concat S6 : 1280 + 640  = 1920  -> 1280, H/32 -> upsample 到 H/16

decoder[6]  concat S5 : 1280 + 640  = 1920  -> 640, H/16
decoder[7]  concat S4 : 640  + 640  = 1280  -> 640, H/16
decoder[8]  concat S3 : 640  + 320  = 960   -> 640, H/16 -> upsample 到 H/8

decoder[9]  concat S2 : 640  + 320  = 960   -> 320, H/8
decoder[10] concat S1 : 320  + 320  = 640   -> 320, H/8
decoder[11] concat S0 : 320  + 320  = 640   -> 320, H/8
        """
        self.decoder=nn.ModuleList(
            [

                SwitchSequential(UNET_Residual(2560,1280)),
                SwitchSequential(UNET_Residual(2560,1280)),
                SwitchSequential(UNET_Residual(2560,1280),Upsample(1280)),

                SwitchSequential(UNET_Residual(2560,1280),UNET_Attn(8,160)),
                SwitchSequential(UNET_Residual(2560,1280),UNET_Attn(8,160)),

                SwitchSequential(UNET_Residual(1920,1280),UNET_Attn(8,160),Upsample(1280)),
                SwitchSequential(UNET_Residual(1920,640),UNET_Attn(8,80)),
                SwitchSequential(UNET_Residual(1280,640),UNET_Attn(8,80)),
                SwitchSequential(UNET_Residual(960,640),UNET_Attn(8,80),Upsample(640)),

                SwitchSequential(UNET_Residual(960,320),UNET_Attn(8,40)),
                SwitchSequential(UNET_Residual(640,320),UNET_Attn(8,40)),
                SwitchSequential(UNET_Residual(640,320),UNET_Attn(8,40)),



            ]
        )


    def forward(self,latent,context,time):
        #x (b,4,h/8,w/8)
        #context (b,seq_len,dim)
        #time(1,1280)
        skip_con=[]
        for layer in self.encoder:
            x=layer(x,context,time)
            skip_con.append(x)

        x=self.bottleneck(x,context,time)

        for layer in self.decoder:
            x=torch.cat((x,skip_con.pop()),dim=1)
            x=layer(x,context,time)


        return x


class Diffusion(nn.Module):
    def __init__(self) :
        super().__init__()
        self.time_emb=TimeEmbedding(320)
        self.unet=UNET()
        self.final=UNET_Outlayer(320,4)


    def forward(self,latent,context,time):
        """
        latent :(b,4,h/8,w/8)
        time :(1,320)
        context: (b,seq_len,dim)

        """
        #(1,320)-->(1,1280)
        
        time=self.time_emb(time)
        #(b,4,h/8,w/8)-->(b,320,h/8,w/8)
        output=self.unet(latent,context,time)
        #(b,320,h/8,w/8)-->(b,4,h/8,w/8)
        output=self.final(output)

        #(b,4,h/8,w/8)
        return output