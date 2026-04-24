import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod



"""
input x_t : (b,c,h,w)
t:        : (b)
unet(x_t,t)->noise :(b,c,h,w)
"""
class EmbedBlock(nn.Module):
    @abstractmethod
    def forward(self,x:torch.Tensor,emb:torch.Tensor)->torch.Tensor:

        raise NotImplementedError
    
class EmbedSequential(nn.Sequential,EmbedBlock):
    def forward(self, x:torch.Tensor,emb:torch.Tensor)->torch.Tensor:
        for layer in self:
            if isinstance(layer,EmbedBlock):
                x=layer(x,emb)
            else:
                x=layer(x)
        return x
    

def timestep_emb(timestep:torch.Tensor,dim:int)->torch.Tensor:
    """
    t->t_emb
    t : (b)
    t_emb: (b,dim)

    freqs=(half)

    t:(b)->(b,1)
    (freqs)->(1,half)
    (b,1)*(1,half)->(b,half)

    cat[cos,sin]->(b,half*2)
    """

    #use cosine,sin position emb
    half=dim//2
    device=timestep.device

    #sin(pos/freqs)
    #where freqs=10000**(-2*i/dmodel)


    freqs=torch.exp(
        -math.log(10000)*torch.arange(half,device=device).float()/half
    )

    #pos (postion)
    #compute the pos/freqs
    args=timestep.float().unsqueeze(1)*freqs.unsqueeze(0)
    #timestep 1 d tensor use unsqueeze(1)
    #(b)-->(b,1)
    #so,freqs 1d 
    #(const)-->(0,const)
    emb=torch.cat([torch.sin(args),torch.cos(args)],dim=1)

    if dim%2==1:
        emb=F.pad(emb,(0,1))
        #处理dim是奇数,最后补个0
    return emb


class Silu(nn.Module):
    def forward(self,x):
        return x*torch.sigmoid(x)


"""
in_c,
out_c,
emb_c,

"""

class Resblock1(nn.Module):
    def __init__(self, in_c:int,out_c:int,emb_c:int,dropout:float=0.0):
        super().__init__()

        self.in_c=in_c
        self.out_c=out_c
        self.emb_c=emb_c

        self.norm1=nn.GroupNorm(32 if in_c>=32 else 1,in_c)
        self.act1=Silu()
        self.conv1=nn.Conv2d(in_c,out_c,3,padding=1)

        self.emb_proj=nn.Sequential(
            Silu(),
            nn.Linear(emb_c,out_c)
        )

        self.norm2=nn.GroupNorm(32 if in_c>=32 else 1,out_c)
        self.act2=Silu()
        self.conv2=nn.Conv2d(out_c,out_c,3,padding=1)

        if in_c!=out_c:
            #solve the shortcut dim diff
            self.skip=nn.Conv2d(in_c,out_c,kernel_size=1)
        else:
            self.skip=nn.Identity()
        
    def forward(self,x:torch.Tensor,emb:torch.Tensor):
        h=self.conv1(self.act1(self.norm1(x)))
        """
        emb:(b,emb_c)
        self.proj(emb)->(b,out_c)
        feature map(b,out_c,h,w)
        self.proj+feature map 
        use [:,:,none,none]
        broadcast -->4dim +4dim
        """
        emb_out=self.emb_proj(emb)[:,:,None,None]
        #(b,out_c,1,1)
        h=h+emb_out
        h=self.conv2(self.act2(self.norm2(h)))

        return h+self.skip(x)
"""
更加只能,不用写很多判断,比如resblock(x,emb),attn(x)
使用embedblock后再embsequential,自动判断
block = EmbedSequential(
    ResBlock(64, 64, emb_c),   # 需要 emb
    AttentionBlock(64),        # 不需要 emb
    Downsample(64),            # 不需要 emb
)
调用
x = block(x, emb)
实际
x = ResBlock(x, emb)
x = AttentionBlock(x)
x = Downsample(x)

"""
    

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
        #(b,c,h,w)->(b,c,2h,2w)


        return self.conv(x)


class BasicUnet(nn.Module):
    def __init__(self, in_c=3,base_c=64,out_c=3):
        super().__init__()

        emb_c=base_c*4

        #time_emb
        self.time_mlp=nn.Sequential(
            nn.Linear(base_c,emb_c),
            Silu(),
            nn.Linear(emb_c,emb_c)

        )

        self.in_cov=nn.Conv2d(in_c,base_c,3,padding=1)
        self.down1=Resblock1(base_c,base_c,emb_c)
        self.down2=Resblock1(base_c,base_c*2,emb_c)

        self.downsample1=Downsample(base_c*2)

        self.down3=Resblock1(base_c*2,base_c*2,emb_c)
        self.down4=Resblock1(base_c*2,base_c*4,emb_c)

        self.downsample2=Downsample(base_c*4)

        #midchannel
        self.mid1=Resblock1(base_c*4,base_c*4,emb_c)
        self.mid2=Resblock1(base_c*4,base_c*4,emb_c)

        #up
        self.upsample1=Upsample(base_c*4)
        self.up1=Resblock1(2*base_c*4,base_c*2,emb_c)
        self.up2=Resblock1(2*base_c*4,2*base_c,emb_c)

        self.upsample2=Upsample(base_c*2)
        self.up3=Resblock1(2*base_c*2,base_c,emb_c)
        self.up4=Resblock1(2*base_c,base_c,emb_c)

        self.out_norm=nn.GroupNorm(32 if base_c>=32 else 1,base_c)

        self.out_act=Silu()
        self.out_conv=nn.Conv2d(base_c,out_c,3,padding=1)

        self.base_c=base_c

    def forward(self,x:torch.Tensor,t:torch.Tensor)->torch.Tensor:
        emb=timestep_emb(t,self.base_c)
        emb=self.time_mlp(emb)

        x=self.in_cov(x)

        #down
        h1=self.down1(x,emb)
        h2=self.down2(h1,emb)
        x=self.downsample1(h2)


        #use concat to add shortcut
        #cat[h,x]
        h3 = self.down3(x, emb)
        h4 = self.down4(h3, emb)
        x = self.downsample2(h4)

        # middle
        x = self.mid1(x, emb)
        x = self.mid2(x, emb)

        # up
        x = self.upsample1(x)
        x = torch.cat([x, h4], dim=1)
        x = self.up1(x, emb)

        x = torch.cat([x, h3], dim=1)
        x = self.up2(x, emb)

        x = self.upsample2(x)
        x = torch.cat([x, h2], dim=1)
        x = self.up3(x, emb)

        x = torch.cat([x, h1], dim=1)
        x = self.up4(x, emb)
        x=self.out_conv(self.out_act(self.out_norm(x)))

        return x
    

class ResBlock(EmbedBlock):
    def __init__(self,in_c:int,out_c:int,emb_c:int,dropout:float =0.0 ,
                 use_scale_shift_norm:bool=False) :
        super().__init__( )
        self.in_c=in_c
        self.out_c=out_c
        self.use_shift_norm=use_scale_shift_norm

        self.in_layers=nn.Sequential(
            nn.GroupNorm(32 if in_c>=32 else 1,in_c),
            Silu(),
            nn.Conv2d(in_c,out_c,3,padding=1),
        )

        self.emb_layers=nn.Sequential(
            Silu(),
            nn.Linear(emb_c,2*out_c if use_scale_shift_norm else out_c),
        )
        self.out_norm=nn.GroupNorm(32 if out_c>=32 else 1,out_c)
        self.out_layers=nn.Sequential(
            
            Silu(),
            nn.Dropout(dropout),
            nn.Conv2d(out_c,out_c,3,padding=1),
        )

        #skip dim
        if in_c!=out_c:
            self.skip=nn.Conv2d(in_c,out_c,1)
        else:
            self.skip=nn.Identity()
    
    def forward(self,x:torch.Tensor,emb:torch.Tensor)->torch.Tensor:
        h=self.in_layers(x)
        emb_out=self.emb_layers(emb)
        while len(emb_out.shape)<len(h.shape):
            emb_out=emb_out[...,None]
        
        if self.use_shift_norm:
            scale,shift=torch.chunk(emb_out,2,dim=1)
            h=self.out_norm(h)*(1+scale)+shift
            h=self.out_layers(h)
        else:
            h=h+emb_out
            h=self.out_layers(self.out_norm(h))
        
        return self.skip(x)+h
    

class AttentionBlock(nn.Module):
    def __init__(self, channel:int,num_heads:int=1) :
        super().__init__()
        assert channel%num_heads==0 ,"this must be int"
        self.channel=channel
        self.num_heads=num_heads

        self.norm=nn.GroupNorm(32 if channel>=32 else 1,channel)

        self.qkv=nn.Conv1d(channel,3*channel,1)
        self.proj=nn.Conv1d(channel,channel,1)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        b,c,h,w=x.shape
        x_in=x
        x=x.reshape(b,c,h*w)
        x=self.norm(x)

        qkv=self.qkv(x)
        #(b,3c,h*w)
        q,k,v=torch.chunk(qkv,3,dim=1)
        #(b,c,h*w)
        head_dim=c//self.num_heads

        q=q.view(b,self.num_heads,head_dim,h*w)
        k=k.view(b,self.num_heads,head_dim,h*w)
        v=v.view(b,self.num_heads,head_dim,h*w)
        """
        q: [B, heads, HW, head_dim]
        now [b,heads,hw,head_dim]
        k: [B, heads, head_dim, HW]
        score: [B, heads, HW, HW]
        消去head_dim
        """
        scale=head_dim**0.5
        score=q.transpose(2,-1)@k
        score=score/scale
        attn=torch.softmax(score,dim=-1)
        #(b,n_heads,h*w,h*w)*   (b,n_heads,dim,h*w).transpose(2,-1)
        attention=attn@v.transpose(2,-1)
        #now (b,n_heads,h*w,dim)
        out=attention.transpose(2,-1).contiguous()
        out=out.view(b,c,h*w)
        out=self.proj(out).reshape(b,c,h,w)

        return x_in+out
    


class BetterUnet(nn.Module):
    def __init__(self, in_c=3,base_c=64,out_c=3,
                 num_class:int |None=None,
                 use_scale_shift_norm:bool=True) :
        super().__init__()

        emb_c=base_c*4
        self.base_c=base_c

        self.time_mlp=nn.Sequential(
            nn.Linear(base_c,emb_c),
            Silu(),
            nn.Linear(emb_c,emb_c)
        )

        self.side_emb=(
            nn.Embedding(num_class,emb_c)
            if num_class is not None else None
        )

        self.in_conv=nn.Conv2d(in_c,base_c,3,padding=1)

        self.down1=EmbedSequential(
            ResBlock(base_c,base_c,emb_c,use_scale_shift_norm=use_scale_shift_norm)
        )

        self.down2=EmbedSequential(
            ResBlock(base_c,base_c*2,emb_c,use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(base_c*2)
        )
        self.downsample=Downsample(base_c*2)

        self.mid=EmbedSequential(
            ResBlock(base_c*2,base_c*4,emb_c,use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(base_c*4),
            ResBlock(4*base_c,4*base_c,emb_c,use_scale_shift_norm=use_scale_shift_norm)
        )

        self.upsample1=Upsample(base_c*4)

        #对应加上self.down2的out channel,因为是concat的self.down2
        self.up1=EmbedSequential(
            ResBlock(base_c*4+base_c*2,base_c*2,emb_c,use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(2*base_c)
        )
        #对应加上self.down1的out channel,....
        self.up2=EmbedSequential(
            ResBlock(2*base_c+base_c,base_c,emb_c,use_scale_shift_norm=use_scale_shift_norm),
            

        )

        self.out=nn.Sequential(
            nn.GroupNorm(32 if base_c>=32 else 1,base_c),
            Silu(),
            nn.Conv2d(base_c,out_c,padding=1)
        )

    def forward(self,x:torch.Tensor,t:torch.Tensor,side:torch.Tensor |None=None):
        emb=timestep_emb(t,self.base_c)
        emb=self.time_mlp(emb)

        if self.side_emb is not None:
            if side is None:
                raise ValueError("side is required")
            emb=emb+self.side_emb(side.long())

        
        x=self.in_conv(x)

        h1=self.down1(x,emb)
        h2=self.down2(h1,emb)
        x=self.downsample(h2)

        x=self.mid(x,emb)

        x=self.upsample1(x)
        x=torch.cat([x,h2],dim=1)
        x=self.up1(x,emb)

        x=torch.cat([x,h2],dim=1)
        x=self.up2(x,emb)

        return self.out(x)