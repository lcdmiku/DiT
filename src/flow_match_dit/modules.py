from typing import Any

import torch
import torch.nn as nn
import math
import einops
from torch.nn import Sequential

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class MLP(nn.Module):
    def __init__(self, in_features, out_features,hidden_size=None):
        super(MLP, self).__init__()
        if hidden_size is None:
            hidden_size = out_features
        self.ln = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_features),
        )

    def forward(self, x):
        return self.ln(x)

class RMSNorm(nn.Module):
    def __init__(self, dim:int,eps:float=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = (x * torch.rsqrt(torch.mean(x**2,dim=-1,keepdim=True) + self.eps)).float()
        return (self.gamma * x).type_as(x)

class PatchEmbed(nn.Module):
    """[B,C,H,W] -> [B,N,D]"""
    def __init__(self, in_channels,patch_size=4,hidden_size=256):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B,D,Hp,Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x,Hp,Wp

class UnPatchify(nn.Module):
    """[B,N,D] -> [B,C,H,W]"""
    def __init__(self,out_channels,patch_size=4,hidden_size=256):
        super(UnPatchify, self).__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.proj = nn.Linear(hidden_size,out_channels*patch_size*patch_size)

    def forward(self, x,Hp,Wp):
        B,N,D = x.shape
        C = self.out_channels
        p = self.patch_size
        #->[B,N,c*p*p]
        x = self.proj(x)
        #->[B,C,p,p,Hp,Wp]
        x = x.transpose(1,2).view(B,C,p,p,Hp,Wp)
        #->[B,C,Hp,p,Wp,p]
        x = x.permute(0,1,4,2,5,3).contiguous()
        #->[B,C,H,W]
        x = x.view(B,C,Hp * p,Wp * p)
        return x


class Encoder(nn.Module):
    def __init__(self,hidden_size,time_emb_dim,output_dim,n_heads=4):
        super().__init__()
        self.adLn = AdaLNZero(time_emb_dim,time_emb_dim,hidden_size)

        self.zt_mlp = MLP(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.attn = MultiheadAttention(n_heads=n_heads,emb_dim=hidden_size,input_dim=hidden_size)
        self.out_mlp = MLP(hidden_size, output_dim,hidden_size)

    def forward(self, x,t,conds,mask):
        a1,b1,g1,a2,b2,g2 = self.adLn(x,t,conds,mask)

        z = self.norm1(x) * (1 + g1[:,None,:]) + b1[:,None,:]
        # print(x.shape)
        #Attn
        z = self.attn(z,t,conds,mask)
        x = x + a1[:,None,:] * z
        #MLP
        h = self.norm2(x) * (1 + g2[:,None,:]) + b2[:,None,:]
        h = self.out_mlp(h)
        x = x + a2[:,None,:] * h
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, T: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        position = torch.arange(T).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, output_dim, 2) * (-math.log(10000.0) / output_dim))
        pe = torch.zeros(T, output_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t: torch.Tensor):
        return self.pe[t].reshape(t.shape[0], self.output_dim)

class RoPE(nn.Module):
    """
    head_dim 被拆成两半: 前半作为实部, 后半作为虚部, 两两配对做旋转。
    要求 head_dim % 2 == 0
    """
    def __init__(self, head_dim :int,theta : float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.d_half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0,self.d_half).float()) / self.d_half)

        self.register_buffer('inv_freq', inv_freq,persistent=False)
        self._cache = {}

    def _build_cache(self,N:int,device,dtype):
        key = (N,device,dtype)
        cache = self._cache.get(key)
        if cache is not None:
            return cache

        pos = torch.arange(N,device=device,dtype=dtype) #[N,]
        freqs = pos[:,None] * self.inv_freq.to(device=device,dtype=dtype)[None] #[N,d_helf]
        cos,sin = freqs.cos(), freqs.sin()
        self._cache[key] = (cos,sin)
        return cos,sin

    def forward(self,q:torch.Tensor,k:torch.Tensor):
        """
        :param q: [B,N,n_heads,head_dim]
        :param k: [B,N,n_heads,head_dim]
        :return:
        """
        B,N,H,D = q.shape
        assert D == self.head_dim
        cos,sin = self._build_cache(N,q.device,q.dtype)
        #-> [1,N,1,d_half]
        cos = cos[None,:,None,:]
        sin = sin[None,:,None,:]

        def apply(x):
            # x: [B, N, H, head_dim]
            if self.head_dim % 2 == 1:
                x_rot, x_pass = x[..., :-1], x[..., -1:]  # [..., D-1], [..., 1]
                x1, x2 = x_rot.chunk(2, dim=-1)
                rotated = torch.cat([x1 * cos - x2 * sin,
                                     x1 * sin + x2 * cos], dim=-1)
                return torch.cat([rotated, x_pass], dim=-1)

            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([x1 * cos - x2 * sin,
                              x1 * sin + x2 * cos], dim=-1)

        return apply(q),apply(k)

class MultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, emb_dim: int, input_dim: int) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0
        head_dim = emb_dim // n_heads
        self.head_dim = head_dim
        self.K_W = nn.Parameter(torch.rand(n_heads, input_dim, head_dim))
        self.K_b = nn.Parameter(torch.rand(n_heads, head_dim))
        self.Q_W = nn.Parameter(torch.rand(n_heads, input_dim, head_dim))
        self.Q_b = nn.Parameter(torch.rand(n_heads, head_dim))
        self.V_W = nn.Parameter(torch.rand(n_heads, input_dim, head_dim))
        self.V_b = nn.Parameter(torch.rand(n_heads, head_dim))
        self.O_W = nn.Parameter(torch.rand(n_heads, head_dim, input_dim))
        self.O_b = nn.Parameter(torch.rand(input_dim))
        for W in [self.K_W, self.Q_W, self.V_W, self.O_W]:
            nn.init.xavier_uniform_(W)
        for b in [self.K_b, self.Q_b, self.V_b,self.O_b]:
            nn.init.zeros_(b)
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm([input_dim]),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )
        #RoPE
        self.rope = RoPE(head_dim)
        self.scale = head_dim ** -0.5

    def forward(self, x, t, conds ,mask):
        # _, input_dim, h, w = x.shape
        # x = einops.rearrange(x, "b c h w -> b (h w) c")
        res = x
        # print(x.shape)
        # res = self.norm(x)
        k = einops.einsum(res, self.K_W, "b size ch, n_h ch h_dim -> b size n_h h_dim")
        k = k + self.K_b
        q = einops.einsum(res, self.Q_W, "b size ch, n_h ch h_dim -> b size n_h h_dim")
        q = q + self.Q_b

        #Rope
        q,k = self.rope(q,k)

        qk = einops.einsum(q, k, "batch s1 n_h h_dim, batch s2 n_h h_dim -> batch n_h s1 s2")
        qk = qk * self.scale
        qk = qk.softmax(-1)
        v = einops.einsum(res, self.V_W, "b size ch, n_h ch h_dim -> b size n_h h_dim")
        v = v + self.V_b
        res = einops.einsum(qk, v, "batch n_h size size, batch size n_h h_dim -> batch size n_h h_dim")
        res = einops.einsum(res, self.O_W, "batch size n_h h_dim, n_h h_dim ch -> batch size ch")
        res = res + self.O_b
        # res = res + x
        # res = self.mlp(res) + res
        # res = einops.rearrange(res, "b (h w) c -> b c h w", h=h, w=w)
        return res

class AdaLNZero(nn.Module):
    def __init__(self,cond_dim,time_emb_dim,hidden_size):
        super().__init__()
        self.mlp = MLP(in_features=cond_dim+time_emb_dim,out_features=6 * hidden_size,hidden_size=hidden_size)

        nn.init.zeros_(self.mlp.ln[-1].weight)
        nn.init.zeros_(self.mlp.ln[-1].bias)

    def forward(self, x, t, conds ,mask):
        conds = conds * mask
        cond_and_t = torch.cat([conds,t],dim=1)
        params = self.mlp(cond_and_t)
        return torch.chunk(params,6,dim=-1)

