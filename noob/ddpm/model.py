import torch
import torch.nn as nn
import torch.nn.functional as F
from from_scratch.ddpm.datasets import get_img_shape

class PositionEncoding(nn.Module):
    def __init__(self,seq_len:int,dim :int) :
        super().__init__()

        assert dim%2==0

        pe=torch.zeros(seq_len,dim)
        i_seq=torch.linspace(0,seq_len-1,seq_len)
        j_seq=torch.linspace(0,dim-2,dim//2)

        pos,twoi=torch.meshgrid(i_seq,j_seq)
        pe2i=torch.sin(pos/10000**(twoi/dim))
        pe2i1=torch.cos(pos/10000**(twoi/dim))

        pe=torch.stack((pe2i,pe2i1),2).reshape(seq_len,dim)

        self.emb=nn.Embedding(seq_len,dim)
        self.emb.weight.data=pe
        self.emb.requires_grad_(False)

    def forward(self,t):
        return self.emb(t.to(self.emb.weight.device))



class ResidualBlock(nn.Module):
    def __init__(self,in_c:int,out_c:int) :
        super().__init__()

        self.conv1=nn.Conv2d(in_c,out_c,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(out_c)
        self.actvation1=nn.ReLU()

        self.conv2=nn.Conv2d(out_c,out_c,3,1,1)
        self.bn2=nn.BatchNorm2d(out_c)
        self.actvation2=nn.ReLU(
        )

        if in_c!=out_c:
            self.shortcut=nn.Sequential(nn.Conv2d(in_c,out_c,1),nn.BatchNorm2d(out_c))
        else:
            self.shortcut=nn.Identity()    
    
    def forward(self,x):
        input=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.actvation1(x)

        x=self.conv2(x)
        x=self.bn2(x)

        x+=self.shortcut(input)
        x=self.actvation2(x)

        return x
    

class CNN(nn.Module):
    def __init__(self,n_steps,channels=[10,20,40],pe_dim=10,
                 insert_all_layers=False) :
        super().__init__()
        c,h,w=get_img_shape(
        )
        self.pe=PositionEncoding(n_steps,pe_dim)

        self.pe_linears=nn.ModuleList()
        #nn.emd()只能转为2d,而img--（bs,c,h,w)
        self.allt=insert_all_layers
        if not insert_all_layers:
            self.pe_linears.append(nn.Linear(pe_dim,c))

        self.resdualblock=nn.ModuleList()

        prevc=c

        for channel in channels:
            self.resdualblock.append(ResidualBlock(prevc,channel))
            if insert_all_layers:
                self.pe_linears.append(nn.Linear(pe_dim,prevc))
            else:
                self.pe_linears.append(None)
            
            prevc=channel
        self.out_layer=nn.Conv2d(prevc,c,3,1,1)

    def forward(self,x,t):
        n=t.shape[0]
        t=self.pe(t)
        for mx,mt in zip(self.resdualblock,self.pe_linears):
            if mt is not None:
                pe=mt(t).reshape(n,-1,1,1)
                x=x+pe
        x=self.out_layer(x)


class unetblock(nn.Module):

    def __init__(self,shape,in_c:int,out_c:int,residual=False) :
        super().__init__()
        self.norm=nn.LayerNorm(shape)
        self.conv1=nn.Conv2d(in_c,out_c,3,1,1)
        self.conv2=nn.Conv2d(out_c,out_c,3,1,1)
        self.activation=nn.ReLU()

        self.residual=residual
        #residual bool-->jugde
        #residaulcov --->memory res
        if residual:
            if in_c==out_c:
                self.resconv=nn.Identity()
            else:
                self.resconv=nn.Conv2d(in_c,out_c,1)

    def forward(self,x):
        out=self.norm(x)
        out=self.conv1(out)
        out=self.activation(out)
        out=self.conv2(out)
        if self.residual:
            out+=self.resconv(x)
        out=self.activation(out)

        return out

class UNET(nn.Module):
    def __init__(self,n_steps,channels=[10,20,40,80],pe_dim=10,residual=False) :
        super().__init__()

        c,h,w=get_img_shape()
        nlayers=len(channels)

        hs=[h]
        ws=[w]

        ch=h
        cw=w
        for i in range(nlayers-1):
            ch//=2
            cw//=2
            hs.append(ch)
            ws.append(cw)
        
        self.pe=PositionEncoding(n_steps,pe_dim)

        self.encoders=nn.ModuleList()
        self.decoders=nn.ModuleList()

        self.pe_encoder_en=nn.ModuleList()
        self.pe_encoder_de=nn.ModuleList()

        self.downs=nn.ModuleList()
        self.ups=nn.ModuleList()

        prev_c=c

        for channel,ch,cw in zip(channels[0:-1],hs[0:-1],ws[0:-1]):
            self.pe_encoder_en.append(
                nn.Sequential(
                    nn.Linear(pe_dim,prev_c),
                    nn.ReLU(),
                    nn.Linear(prev_c,prev_c)
                )
            )

            self.encoders.append(
                nn.Sequential(
                    unetblock((prev_c,ch,cw),
                              prev_c,channel,
                              residual=residual),
                    unetblock((channel,ch,cw),
                              channel,channel,
                              residual=residual)    
                )

            )
            self.downs.append(nn.Conv2d(channel,channel,2,2))

            prev_c=channel

            #mid
        
        self.pe_mid=nn.Linear(pe_dim,prev_c)
        #80
        channel=channels[-1]
        self.mid=nn.Sequential(
            unetblock((prev_c,hs[-1],ws[-1]),
                      prev_c,channel,
                      residual=residual
                      ),
            unetblock(
                (channel,hs[-1],ws[-1]),
                channel,channel,
                residual=residual
            )
        )
        #->80  which mean channels[-1]
        prev_c=channel

        for channel ,ch,cw in zip(channels[-2::-1],hs[-2::-1],ws[-2::-1]):
            self.pe_encoder_de.append(
                nn.Linear(pe_dim,prev_c)

            )

            self.ups.append(nn.ConvTranspose2d(prev_c,channel,2,2))

            self.decoders.append(
                nn.Sequential(
                    unetblock((channel*2,ch,cw),
                              channel*2,
                              channel,
                              residual=residual),
                    unetblock((channel,ch,cw),
                              channel,channel,
                              residual=residual)
                )
            )
            prev_c=channel
        
        self.convout=nn.Conv2d(prev_c,c,3,1,1)

    
    def forward(self,x,t):
        device = x.device
        
        n=t.shape[0]
    # 确保 t 也在同一个设备上
        t = t.to(device)
     
        t=self.pe(t).to(device)
        encoders_out=[]

        for pe_linear,encoder,down in zip(self.pe_encoder_en,self.encoders,self.downs):
            pe=pe_linear(t).reshape(n,-1,1,1)
            pe=pe.to(device)
            #add time
            x=encoder(x+pe)
            #memory residaul
            encoders_out.append(x)
            #do downsample
            x=down(x)
        
        pe=self.pe_mid(t).reshape(n,-1,1,1).to(device)
        x=self.mid(x+pe)

        for pe_linear,decoder,up,enocderout in zip(self.pe_encoder_de,self.decoders,self.ups,encoders_out[::-1]
                                        ):
            pe=pe_linear(t).reshape(n,-1,1,1).to(device)
            x=up(x)
            #除法的特性，h/w奇数的整除问题---与纯偶数的矛盾
            #因此需要pad来统一
            pad_x=enocderout.shape[2]-x.shape[2]
            pad_y=enocderout.shape[-1]-x.shape[-1]
            x=F.pad(x,(pad_x//2,pad_x-pad_x//2,pad_y//2,pad_y-pad_y//2))


            #实际上residual是把channel cat了，所以在decoder中channel*2
            #并且在downsample之前存residual的目的是：after img pass downsample ,the imformation of low
            #level loss,so we prefer to memory before down
            x=torch.cat((enocderout,x),dim=1)
            x=decoder(x+pe)
        x=self.convout(x)

        return x

convnet_small_cfg = {
    'type': 'CNN',
    'intermediate_channels': [10, 20],
    'pe_dim': 128
}

convnet_medium_cfg = {
    'type': 'CNN',
    'intermediate_channels': [10, 10, 20, 20, 40, 40, 80, 80],
    'pe_dim': 256,
    'insert_t_to_all_layers': True
}
convnet_big_cfg = {
    'type': 'CNN',
    'intermediate_channels': [20, 20, 40, 40, 80, 80, 160, 160],
    'pe_dim': 256,
    'insert_t_to_all_layers': True
}

unet_1_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}
unet_res_cfg = {
    'type': 'UNet',
    'channels': [10, 20, 40, 80],
    'pe_dim': 128,
    'residual': True
}

def build_network(config:dict,n_steps):
    network_type=config.pop('type')
    if network_type=='CNN':
        network=CNN
    else:
        network=UNET
    
    networks=network(n_steps,**config)
    return networks