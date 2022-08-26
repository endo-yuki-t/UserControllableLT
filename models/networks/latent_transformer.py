import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv, query_length=1):
        h = self.heads

        k = self.to_k(x_qkv)[:, query_length:]
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)[:, query_length:]
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv)[:, :query_length]
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out
        
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 6, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, y):
        x = x + self.pos_embedding[:, :x.shape[1]]
        for sattn, cattn, ff in self.layers:
            x = sattn(x) + x
            xy = torch.cat((x,y), dim=1)
            x = cattn(xy, query_length=x.shape[1]) + x
            x = ff(x) + x
        return x

class Network(nn.Module):
    def __init__(self, opts):
        super(Network, self).__init__()
        
        self.transformer_encoder = TransformerEncoder(dim=512, depth=6, heads=8, dim_head=64, mlp_dim=512, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=512, depth=6, heads=8, dim_head=64, mlp_dim=512, dropout=0)
        self.layer1 = nn.Linear(3, 256)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, 512)
        self.mlp_head = nn.Sequential(
            nn.Linear(512, 512)
        )
         
    def forward(self, w, x, y, alpha=1.):
        #w: latent vectors
        #x: flow vectors
        #y: StyleGAN features
        xh = F.relu(self.layer1(x))
        yh = F.relu(self.layer2(y))
        xyh = torch.cat([xh,yh], dim=2)
        xyh = F.relu(self.layer3(xyh))
        xyh = self.transformer_encoder(xyh)
        
        wh = F.relu(self.layer4(w))
        
        h = self.transformer_decoder(wh, xyh)
        h = self.mlp_head(h)
        w_hat = w+alpha*h
        return w_hat
