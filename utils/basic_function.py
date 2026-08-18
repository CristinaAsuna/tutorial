import torch
from torch import nn
from torch.nn import functional as F

class Residualblock(nn.Module):
    def __init__(self, inc, outc, dropout=0.0):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, inc, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, outc, eps=1e-6)

        self.conv1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv2d(inc, outc, kernel_size=1)
            if inc != outc else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)

        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))

        return x + residual
    
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