import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
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
        residual=self.residual(x)
        x=self.conv1(F.silu(self.norm1(x)))
        x=self.conv2(F.silu(self.norm2(x)))
        

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
        residual=x
        is_spatial=x.ndim==4
        if is_spatial:
            b,c,h,w=x.shape
            x=x.flatten(2).transpose(1,2)
         # 此后的原始注意力逻辑保持不变

        initial_shape = x.shape
        b, seq_len, dmodel = x.shape
        split_shape = (b, seq_len, self.nheads, self.ndim)

        q, k, v = self.inlinear(x).chunk(3, dim=-1)
        q = q.view(split_shape).transpose(1, 2)
        k = k.view(split_shape).transpose(1, 2)
        v = v.view(split_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        score = weight * self.scale
        score = F.softmax(score, dim=-1)
        attn = score @ v

        out = attn.transpose(1, 2)
        out = out.contiguous().reshape(initial_shape)
        out = self.out(out)

        # (B, H*W, C) -> (B, C, H, W)，让后续 Conv2d 能继续处理
        if is_spatial:
            out = out.transpose(1, 2).reshape(b, c, h, w)

        return out+residual
    
class AEencoder(nn.Module):
    def __init__(self,inc=1,factor=8,latent=64):
        super().__init__()
        assert factor > 0 and factor & (factor - 1) == 0, "factor 必须是 2 的幂，例如 2、4、8"
        n_down = factor.bit_length() - 1

        layers = [
            nn.Conv2d(inc, 128, kernel_size=3, padding=1),
            Residualblock(128, 128),
            Residualblock(128, 128),
        ]

        # factor=4 时循环 2 次；factor=8 时循环 3 次。
        # down_channels 记录每次下采样前的通道数，decoder 用它对称地恢复通道。
        self.down_channels = []
        channels = 128
        for i in range(n_down):
            self.down_channels.append(channels)
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))
            if i < n_down - 1:
                next_channels = min(channels * 2, 512)
                layers.extend([
                    Residualblock(channels, next_channels),
                    Residualblock(next_channels, next_channels),
                ])
                channels = next_channels

        layers.extend([
            Residualblock(channels, channels),
            Residualblock(channels, channels),
            Residualblock(channels, channels),
            MHSA(dmodel=channels, nheads=2),
            Residualblock(channels, channels),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, latent, kernel_size=3, padding=1),
        ])
        self.encode = nn.ModuleList(layers)
    def forward(self,x):
        for layer in self.encode:
            x=layer(x)
        return x
    
class AEdecoder(nn.Module):
    def __init__(self,inc,factor=8,latent=64):
        super().__init__()
        assert factor > 0 and factor & (factor - 1) == 0, "factor 必须是 2 的幂，例如 2、4、8"
        n_down = factor.bit_length() - 1

        down_channels = []
        channels = 128
        for i in range(n_down):
            down_channels.append(channels)
            if i < n_down - 1:
                channels = min(channels * 2, 512)

        layers = [
            nn.Conv2d(latent, latent, kernel_size=1),
            nn.Conv2d(latent, channels, kernel_size=3, padding=1),
            Residualblock(channels, channels),
            MHSA(channels, nheads=2),
            Residualblock(channels, channels),
            Residualblock(channels, channels),
            Residualblock(channels, channels),
            Residualblock(channels, channels),
        ]

        # 按 encoder 的相反顺序上采样，factor=4 时恰好上采样两次。
        for target_channels in reversed(down_channels):
            layers.extend([
                nn.Upsample(scale_factor=2),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            ])
            if channels != target_channels:
                layers.append(Residualblock(channels, target_channels))
                channels = target_channels
            layers.extend([
                Residualblock(channels, channels),
                Residualblock(channels, channels),
            ])

        layers.extend([
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, inc, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ])
        self.decode = nn.ModuleList(layers)
    def forward(self,x):
        for layer in self.decode:
            x=layer(x)
        return x

class AE(nn.Module):
    def __init__(self, encoder:AEencoder,decoder:AEdecoder) :
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
    
    def forward(self,x):
        latent=self.encoder(x)
        out=self.decoder(latent)
        return  out

# def train(model,device,epochs):
    
#     for epoch in epochs:
#         x=dataloader.iter

#         latents=AEencoder(x)
#         out=AEdecoder(latents)

#         loss=F.mse_loss(x,out).sum(dim=0)
#         loss.backward()


def train_autoendoer(model,dataloader,
                     epochs=10,lr=1e-3,
                     device="cpu",
                     ):

    device=torch.device(device if torch.cuda.is_available() else "cpu")
    model=model.to(device)

    optimizer=AdamW(model.parameters(),lr=lr)
    criterion=nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss=0.0
        total_samples=0
        progress_bar=tqdm(
            dataloader,
            desc=f"epoch {epoch+1}/{epochs}",
            unit="batch",
        )

        for batch in progress_bar:
            x=batch[0] if isinstance(batch,(tuple,list)) else batch
            x=x.to(device,non_blocking=True)

            optimizer.zero_grad()
            reconstructed=model(x)
            loss=criterion(reconstructed,x)

            loss.backward()
            optimizer.step()

            batch_size=x.size(0)
            total_loss+=loss.item()*batch_size
            total_samples+=batch_size

            progress_bar.set_postfix(
                loss=f"{loss.item():.5f}",
                avg_loss=f"{total_loss/total_samples:.5f}",
            )

        avg_loss=total_loss/total_samples
        print(f"Epoch [{epoch + 1}/{epochs}] | MSE Loss: {avg_loss:.6f}")
    return model

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    # batch,h,w=2,64,64
    # x=torch.randn(batch,1,h,w,device=device)

    # model=AE(encoder=AEencoder(inc=1,latent=64),
    #          decoder=AEdecoder(inc=1,latent=64)).to(device)

    # model.eval()
    # with torch.no_grad():
    #     y=model(x)
    transform=transforms.ToTensor()
    train_dataset=datasets.MNIST(
        root=r"D:\Datasets",
        train=True,
        transform=transform,
        download=False,
    )
    train_loader=DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,

    )
    factor=4
    model=AE(
        encoder=AEencoder(inc=1,factor=factor,latent=64),
        decoder=AEdecoder(inc=1,factor=factor,latent=64),

    )
    model=train_autoendoer(
        model=model,
        dataloader=train_loader,
        epochs=2,
        lr=1e-3,
        device=device,
    )
    torch.save(model.state_dict(),"mnist_ae.pth")
    total=sum(p.numel() for p in model.parameters())
    print(f"model size: {total:,}")

if __name__=="__main__":
    main()

