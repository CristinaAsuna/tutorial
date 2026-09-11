import torch
from torch import nn
from torch.nn import functional as F
from attention import MHSA

"""
clip
    input x :(b,seq_len)
    output : (b,seq_len,dim)

    clipembed
    cliplayer

"""
class Clip(nn.Module):
    def __init__(self) :
        super().__init__()

        self.emb=ClipEmbedding(49408,768,77)
        self.layers=nn.ModuleList(
            [ClipLayer(12,768) for i in range(12)]
        )
        self.layernorm=nn.LayerNorm(768)

    def forward(self,tokens:torch.LongTensor)->torch.FloatTensor:
        tokens=tokens.type(torch.long)

        #emb (b,seq_len)->(b,seq_len,dim)
        state=self.emb(tokens)

        #layers similar to transformer encoder
        #(b,seq_len,dim)->(b,seq_len,dim)
        for layer in self.layers:
            state=layer(state)
        
        output=self.layernorm(state)

        return output
    
class ClipEmbedding(nn.Module):
    def __init__(self,vocab_size,dim,seq_len) :
        super().__init__()

        self.token_emb=nn.Embedding(vocab_size,dim)
        self.pos_emb=nn.Parameter(torch.zeros(seq_len,dim))

    def forward(self,tokens):
        x=self.token_emb(tokens)
        x+=self.pos_emb
        return x
    
#transformer encoder
class ClipLayer(nn.Module):
    def __init__(self,n_heads,dim) :
        super().__init__()

        self.norm1=nn.LayerNorm(dim)
        self.attn=MHSA(num_heads=n_heads,dim=dim)
        #ffn
        self.ln1=nn.Linear(dim,4*dim)
        self.ln2=nn.Linear(4*dim,dim)
        self.norm2=nn.LayerNorm(dim)

    def forward(self,x):
        residaul=x
        #x (b,seq_len,dim)
        x=self.norm1(x)
        x=self.attn(x,causal_mask=True)
        x+=residaul

        #ffn
        residual=x
        x=self.norm2(x)
        x=self.ln1(x)
        #use quick gelu
        x=x*torch.sigmoid(1.702*x)
        x=self.ln2(x)

        x+=residual

        return x



