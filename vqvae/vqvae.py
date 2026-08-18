import torch
import sys
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# 支持从 vqvae 目录直接运行本文件。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vae.vae import VAEencoder,VAEdecoder

# class VectorQuantizerEMA(num_codes=256,code_dim=32):
    
#     pass


class VQ_VAE(nn.Module):
     def __init__(self,inc=1,factor=8,latent=32,double_z=False,
                  code_dim=32,num_codes=256):
        super().__init__()

        self.encoder=VAEencoder(inc=inc,factor=factor,
                                latent=latent,double_z=double_z)
        
        self.quantizer=VectorQuantizer(
            num_codes=num_codes,
            code_dim=code_dim
        )

        self.decoder=VAEdecoder(inc=inc,factor=factor,
                                latent=latent)
        
     def forward(self,x):
         ze=self.encoder(x)
         zq,indices,vq_loss,quant_metrics=self.quantizer(ze)
         zq_st=ze+(zq-ze).detach()
         x_hat=self.decoder(zq_st)
         
         return x_hat,indices,vq_loss










class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=256, code_dim=32, beta=0.25):
        super().__init__()
        """
    return 
        z_q
        indices
        vq_loss
    """
    #parameterize codebook,(k,d) tensor
        self.num_codes=num_codes
        self.code_dim=code_dim
        self.embedding=nn.Embedding(num_codes,code_dim)
        nn.init.uniform_(self.embedding.weight,-1/num_codes,1/num_codes)
        self.beta=beta

    def forward(self,z_e):
        #z_e (b,c,h,w)-->(b,h,w,c)-->(b*w*h,c)
        b,c,h,w=z_e.shape
        if c != self.code_dim:
            raise ValueError(f"code_dim 应为 {self.code_dim}，实际为 {c}")
        z_flat=z_e.permute(0,2,3,1).contiguous().view(-1,c)

        codebook=self.embedding.weight

        #compute distance
        #(b*h*w,k)
        distance=(z_flat.pow(2).sum(dim=1,keepdim=True)
                  +codebook.pow(2).sum(dim=1)
                  -2*z_flat@codebook.t())
        
        # find index
        #(b*h*w,)
        flat_indices=distance.argmin(dim=1)
        # search by index,return zq
        #(b*h*w,dim)
        #z_flat[i] 被 codebook[flat_indices[i]] 替换
        z_q=self.embedding(flat_indices)

        #reshape z_q--->(b,dim,h,w),where dim=c,-->(b,c,h,w)
        z_q=z_q.view(b,h,w,c).permute(0,3,1,2).contiguous()
        # permute 就是当前位置放前面的index

        #loss
        #但认为encoder,embedding学习速率不一样,将其拆分
        #ze <---> zq, close
        # codebook loss 只更新 embedding；commitment loss 只更新 encoder。
        codebook_loss=F.mse_loss(z_q,z_e.detach())
        commitment_loss=F.mse_loss(z_e,z_q.detach())
        vq_loss=codebook_loss + self.beta*commitment_loss

        indices=flat_indices.view(b,h,w)

        counts=torch.bincount(flat_indices,minlength=self.num_codes).float()
        probs=counts/counts.sum().clamp_min(1.0)
        perplexity=torch.exp(-(probs*torch.log(probs+1e-10)).sum())
        metrics={
            "vq":vq_loss.detach(),
            "codebook":codebook_loss.detach(),
            "commitment":commitment_loss.detach(),
            "perplexity":perplexity.detach(),
            "active_codes":(counts>0).sum().detach(),
        }

        return z_q,indices,vq_loss,metrics

class VQVAE(nn.Module):
     def __init__(self,inc=1,factor=8,latent=32,double_z=False,
                  code_dim=32,num_codes=256):
        super().__init__()

        self.encoder=VAEencoder(inc=inc,factor=factor,
                                latent=latent,double_z=double_z)
        
        self.quantizer=VectorQuantizer(
            num_codes=num_codes,
            code_dim=code_dim
        )

        self.decoder=VAEdecoder(inc=inc,factor=factor,
                                latent=latent)
        
     def forward(self,x):
         ze=self.encoder(x)
         zq,indices,vq_loss,quant_metrics=self.quantizer(ze)
         zq_st=ze+(zq-ze).detach()
         x_hat=self.decoder(zq_st)
         reconstruct_loss=F.mse_loss(x_hat,x)
         total_loss=reconstruct_loss+vq_loss

         metrics={
             "total":total_loss.detach(),
             "recon":reconstruct_loss.detach(),
             **quant_metrics,
         }
         return x_hat,indices,total_loss,metrics


def train_vqvae(model,dataloader,epochs=20,lr=3e-4,device="cpu"):
    """训练普通 VQ-VAE：重建损失 + VQ codebook/commitment loss。"""
    device=torch.device(device if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    optimizer=AdamW(model.parameters(),lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss=0.0
        total_samples=0
        progress_bar=tqdm(dataloader,desc=f"Epoch {epoch+1}/{epochs}",unit="batch")

        for batch in progress_bar:
            x=batch[0] if isinstance(batch,(tuple,list)) else batch
            x=x.to(device,non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            _,_,loss,metrics=model(x)
            loss.backward()
            optimizer.step()

            batch_size=x.size(0)
            total_loss+=metrics["total"].item()*batch_size
            total_samples+=batch_size
            progress_bar.set_postfix(
                total=f"{metrics['total'].item():.5f}",
                recon=f"{metrics['recon'].item():.5f}",
                vq=f"{metrics['vq'].item():.5f}",
                ppl=f"{metrics['perplexity'].item():.1f}",
                active=int(metrics["active_codes"].item()),
            )

        print(f"Epoch [{epoch+1}/{epochs}] | loss={total_loss/total_samples:.6f}")

    return model


def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    factor=4
    code_dim=32
    num_codes=256

    dataset=datasets.MNIST(
        root=r"D:\Datasets",
        train=True,
        transform=transforms.ToTensor(),
        download=False,
    )
    dataloader=DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model=VQVAE(
        inc=1,
        factor=factor,
        latent=code_dim,
        code_dim=code_dim,
        num_codes=num_codes,
    )
    model=train_vqvae(model,dataloader,epochs=5,lr=3e-4,device=device)

    save_path="mnist_vqvae_f4_d32_k256.pth"
    torch.save(model.state_dict(),save_path)
    print(f"Saved: {save_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__=="__main__":
    main()
    
