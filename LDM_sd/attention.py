import torch
from torch import nn
from torch.nn import functional as F

"""
(b,seq_len,dim)-->(b,seq_len,dim)

(b,num_heads,seq_len,d_heads)@(b,num_heads,d_heads,seq_len)
--->(b,num_heads,seq_len,seq_len)-->softmax(dim=-1)
(b,num_heads,seq_len,seq_len)@(b,num_heads,seq_len,d_heads)
--->(b,num_heads,seq_len,d_heads)



"""

class MHSA(nn.Module):
    def __init__(self, num_heads,dim,in_proj_bias=True,out_proj_bias=True):

        super().__init__()
        assert dim%num_heads==0,"dim%n_heads supposed to be int"

        self.n_heads=num_heads
        self.dim=dim
        self.d_head=dim//num_heads
        
        self.in_proj=nn.Linear(dim,dim*3,bias=in_proj_bias)
        self.out_proj=nn.Linear(dim,dim,bias=out_proj_bias)
    
    def forward(self,x,causal_mask=False):
        initial_shape=x.shape
        (b,seq_len,dim)=initial_shape
        heads_shape=(b,seq_len,self.n_heads,self.d_head)

        #(b,seq_len,dim)->(b,seq_len,3*dim)
        q,k,v=self.in_proj(x).chunk(3,dim=-1)
        #(b,seq_len,dim)->(b,seq_len,n_heads,d_heads)
        #-->transpose (b,n_heads,seq_len,d_heads)
        q=q.view(heads_shape).transpose(1,2)
        k=k.view(heads_shape).transpose(1,2)
        v=v.view(heads_shape).transpose(1,2)


        scale=self.d_head**-0.5
        weight=q@k.transpose(-1,-2)

        if causal_mask:
            mask=torch.ones_like(weight,dtype=torch.bool).triu(1)
            weight.masked_fill_(mask,-torch.inf)

        weight=weight*scale
        weight=F.softmax(weight,dim=-1)
        attn=weight@v

        output=attn.transpose(1,2)
        output=output.contiguous().reshape(initial_shape)

        output=self.out_proj(output)
        return output