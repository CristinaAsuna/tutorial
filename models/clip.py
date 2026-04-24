import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import VIT,MHA,FFN,TransformerEncoderblock

class TextEncoder(nn.Module):
    def __init__(self,
                 vocab_size:int,
                 seq_len:int,
                 context_len:int=512,
                 layers:int=6,
                 nheads:int=8,
                 emb_dim:int=512):
        super().__init__()
        self.context_len=context_len
        self.seq_len=seq_len

        self.token_emb=nn.Embedding(vocab_size,context_len)
        self.pos_emb=nn.Parameter(
            torch.randn(1,seq_len,context_len)
        )
        self.transformer=nn.ModuleList(
            [TransformerEncoderblock(context_len,nheads)] for _ in range(layers)
        )

        self.ln=nn.LayerNorm(context_len)

        self.poj=nn.Linear(context_len,emb_dim)

    def forward(self,text:torch.Tensor)->torch.Tensor:
        # text (b,seq_len)

        x=self.token_emb(text)
        #->(b,se_len,context)
        x=x+self.pos_emb[:, :x.shape[1], :]
        #choose the same lenght of x.shape[1]

        for block in self.transformer:
            x=block(x)
        
        x=self.ln(x)
        #(b,context)
        x=x.mean(dim=1)
        x=self.poj(x)
        return x
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.proj(x)      # [B, D, Gh, Gw]
        x = x.flatten(2)      # [B, D, N]
        x = x.transpose(1, 2) # [B, N, D]
        return x   
class VisionEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=width,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, width))

        self.transformer = nn.ModuleList([
            TransformerEncoderblock(width, heads) for _ in range(layers)
        ])

        self.ln_post = nn.LayerNorm(width)
        self.image_projection = nn.Linear(width, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, 224, 224]
        """
        B = images.shape[0]

        x = self.patch_embed(images)   # [B, N, width]

        cls = self.cls_token.expand(B, -1, -1)   # [B,1,width]
        x = torch.cat([cls, x], dim=1)           # [B,N+1,width]

        x = x + self.pos_embedding[:, :x.shape[1], :]  # [B,N+1,width]

        for block in self.transformer:
            x = block(x)   # [B,N+1,width]

        x = self.ln_post(x)     # [B,N+1,width]
        x = x[:, 0]             # [B,width] ȡ cls token
        x = self.image_projection(x)  # [B,embed_dim]
        return x
    

#first to do the normlize of Text,image emb individly
#logits=scale*img_vec@text_vec.transport()

class Clip(nn.Module):
    def __init__(self,
                 vocab_size:int,
                  seq_len:int,
                   emb_dim:int,
                    img_size:int,
                     patch_size:int ):
        super().__init__()

        self.img_encoder=VisionEncoder(
            img_size=img_size,
            patch_size=patch_size,
            width=768,
            layers=12,
            heads=12,
            embed_dim=emb_dim
        )

        self.txt_encoder=TextEncoder(
            vocab_size=vocab_size,
            seq_len=seq_len,
            context_len=512,
            layers=12,
            nheads=8,
            emb_dim=emb_dim
        )
        self.logit_scale=nn.Parameter(
            torch.ones([])*math.log(1/0.07)
        )
    def forward(self,img:torch.Tensor,txt:torch.Tensor):
        img_emb=self.img_encoder(img)
        txt_emb=self.txt_encoder(txt)

        img_norm=F.normalize(img_emb,dim=-1)
        txt_norm=F.normalize(txt_emb,dim=-1)

        scale=self.logit_scale.exp()

        logit_per_img=scale*img_norm@txt_norm.t()
        logit_per_txt=logit_per_img.t()
        #
        return logit_per_img,logit_per_txt
    
def clip_loss(logits_per_img:torch.Tensor,logits_per_txt:torch.Tensor)->torch.Tensor:
    b=logits_per_img.shape[0]

    labels=torch.arange(b,device=logits_per_img.device)

    loss_i=F.cross_entropy(logits_per_img,labels)
    loss_t=F.cross_entropy(logits_per_txt,labels)

    return (loss_i+loss_t)/2

def main():
    model=Clip(vocab_size=50000)
    images = torch.randn(32, 3, 224, 224)
    text_ids = torch.randint(0, 50000, (32, 77))

    logits_per_image, logits_per_text = model(images, text_ids)
    loss = clip_loss(logits_per_image, logits_per_text)

    loss.backward()
    print(loss)
if __name__ =="main":
    main()