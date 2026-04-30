import torch
from torch import nn
from torch.nn import functional as F
import math
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
    
class CrossAttention(nn.Module):
    def __init__(self, num_heads,dim,dim_cross,in_proj_bias=True,out_proj_bias=True):

        super().__init__()
        #q
        self.q=nn.Linear(dim,dim,bias=in_proj_bias)

        #kv cross
        self.k=nn.Linear(dim_cross,dim,bias=in_proj_bias)
        self.v=nn.Linear(dim_cross,dim,bias=in_proj_bias)

        self.out_proj=nn.Linear(dim,dim,bias=out_proj_bias)

        assert dim%num_heads==0,"should be int"
        self.n_heads=num_heads
        self.d_head=dim//num_heads

    def forward(self,x,text):
        #x (b,seq_len_q,dim_q)
        #text (b,seq_len_kv,dim_kv)=(b,77,768)
        init_shape=x.shape
        b,seq_len,dim=init_shape
        temp_shape=(b,seq_len,self.n_heads,self.d_head)

        #->(b,seq_lenq,dim)
        q=self.q(x)
        #kv
        #->(b,seq_lenkv,dim)
        k=self.k(text)
        v=self.v(text)

        q=q.view(temp_shape).transpose(1,2)
        k=k.view(temp_shape).transpose(1,2)
        v=v.view(temp_shape).transpose(1,2)
        ## (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) 
        #-> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        #(b,n_heads,seq_len_q,seq_len_kv)
        weight=q@k.transpose(-1,-2)
        weight/=math.sqrt(self.d_head)
        
        attn=F.softmax(weight,dim=-1)
        out=attn@v
        #(b,n_heads,seq_len_q,d_heads)
        out=out.transpose(1,2).contiguous()
        out=out.view(init_shape)

        #->(b,seq_len_q,dim)
        output=self.out_proj(out)

        return output