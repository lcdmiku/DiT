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
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.ln = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
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

class Encoder(nn.Module):
    def __init__(self,hidden_size,time_emb_dim,num_classes):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.hidden_size = hidden_size

        self.time_mlp = MLP(time_emb_dim, hidden_size)
        self.condition_mlp = MLP(time_emb_dim, hidden_size)

        self.zt_mlp = MLP(hidden_size, hidden_size)

        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

        self.attn = MultiheadAttention(n_heads=4,emb_dim=hidden_size,input_dim=hidden_size)
        self.out_mlp = MLP(hidden_size, hidden_size)

    def forward(self, x,t,conds,mask):
        cond_emb = self.condition_mlp(conds) * mask[:, None]
        time_emb = self.time_mlp(t)
        x = self.zt_mlp(x) + cond_emb + time_emb
        z = self.attn(self.norm1(x),t,conds,mask)
        x = x + z
        h = self.out_mlp(self.norm2(x))
        x = x + h
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
        self.norm = nn.LayerNorm([input_dim])
        self.mlp = nn.Sequential(
            nn.LayerNorm([input_dim]),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, x, t, conds ,mask):
        _, input_dim, h, w = x.shape
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        res = x
        res = self.norm(x)
        k = einops.einsum(res, self.K_W, "b size ch, n_h ch h_dim -> b size n_h h_dim")
        k = k + self.K_b
        q = einops.einsum(res, self.Q_W, "b size ch, n_h ch h_dim -> b size n_h h_dim")
        q = q + self.Q_b
        qk = einops.einsum(q, k, "batch s1 n_h h_dim, batch s2 n_h h_dim -> batch n_h s1 s2")
        qk = qk / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        qk = qk.softmax(-1)
        v = einops.einsum(res, self.V_W, "b size ch, n_h ch h_dim -> b size n_h h_dim")
        v = v + self.V_b
        res = einops.einsum(qk, v, "batch n_h size size, batch size n_h h_dim -> batch size n_h h_dim")
        res = einops.einsum(res, self.O_W, "batch size n_h h_dim, n_h h_dim ch -> batch size ch")
        res = res + self.O_b
        res = res + x
        res = self.mlp(res) + res
        res = einops.rearrange(res, "b (h w) c -> b c h w", h=h, w=w)
        return res

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, is_residual=True, is_debug=False):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        )
        self.conv_2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )
        self.time_emb = MLP(time_emb_dim, out_channels)
        self.condition_emb = MLP(time_emb_dim,out_channels)
        if in_channels != out_channels:
            self.conv_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.conv_3 = nn.Identity()
        self.is_debug = is_debug
        self.is_residual = is_residual

    def forward(self, x, t,conds,mask):
        h = self.conv_1(x)
        if t is None:
            return self.conv_3(x) + self.conv_2(h)
        t = self.time_emb(t)
        conds = self.condition_emb(conds) * mask[:, None]
        batch_size, emb_dim = t.shape
        t = t.view(batch_size, emb_dim, 1, 1)
        conds = conds.view(batch_size, emb_dim, 1, 1)
        if self.is_residual:
            return self.conv_3(x) + self.conv_2(h + t + conds)
        else:
            return self.conv_2(h + t + conds)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim,
                 res_num=2, attn=True, down_sample=True,
                 is_residual=True, is_debug=False):
        super().__init__()
        res_blocks = []
        for i in range(res_num):
            res_blocks.append(ResnetBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                time_emb_dim=time_emb_dim,
                is_residual=is_residual,
            ))
            if attn:
                res_blocks.append(MultiheadAttention(
                    n_heads=4, emb_dim=out_channels, input_dim=out_channels))
        self.res_attn = SequenceWithTimeEmbedding(res_blocks)
        self.downsample = nn.AvgPool2d(2) if down_sample else None

    def forward(self, x, t,conds,mask):
        x = self.res_attn(x, t,conds,mask)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip

class DownBlocks(nn.Module):
    def __init__(self, time_emb_dim,steps=(1,2,4), hidden_size=128, attn_step=None,res_num=3,has_residual=True, is_debug=False):
        super().__init__()
        if attn_step is None:
            attn_step = [1]
        self.down_blocks = nn.ModuleList()
        for (idx,step) in enumerate(steps):
            self.down_blocks.append(
                DownBlock(
                    in_channels=steps[idx-1]*hidden_size if idx > 0 else steps[0] * hidden_size,
                    out_channels=step * hidden_size,
                    time_emb_dim= time_emb_dim,
                    is_residual=has_residual,
                    res_num=res_num,
                    attn = step in attn_step,
                    down_sample = (idx != len(steps) - 1),#最后一个不进行down_sample
                    is_debug = is_debug,
                )
            )

    def forward(self, x, t,conds,mask):
        #收集skip
        skips = []
        for block in self.down_blocks:
            x ,skip= block(x, t,conds,mask)
            if skip is not None:
                skips.append(skip)
        return x , skips


class UpBlock(nn.Module):
    def __init__(self,in_channels, out_channels,time_emb_dim,res_num=2, attn=True,up_sample=True,is_residual=True, is_debug=False):
        super().__init__()
        self.up_block = nn.ModuleList()
        res_blocks = []
        for res_block in range(res_num):
            res_blocks.append(
                ResnetBlock(
                    in_channels=in_channels if res_block == 0 else out_channels,
                    out_channels=out_channels,
                    time_emb_dim=time_emb_dim,
                    is_residual=is_residual,
                )
            )
            # 添加attn
            if attn:
                res_blocks.append(
                    MultiheadAttention(
                        n_heads=4,
                        emb_dim=out_channels,
                        input_dim=out_channels,
                    )
                )
        # 序列化
        self.up_block.append(SequenceWithTimeEmbedding(res_blocks))

        if up_sample:
            self.up_block.append(
                nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2)
            )

    def forward(self, x, t,conds,mask):
        for block in self.up_block:
            if not isinstance(block, SequenceWithTimeEmbedding):
                x = block(x)
            else:
                x = block(x, t,conds,mask)
        return x


class UpBlocks(nn.Module):
    def __init__(self, time_emb_dim, steps=(1, 2, 4), hidden_size=128, attn_step=None, res_num=3, has_residual=True,
                 is_debug=False):
        super().__init__()
        if attn_step is None:
            attn_step = [1]
        self.up_blocks = nn.ModuleList()
        rev_steps = list(reversed(steps))
        for (idx, step) in enumerate(rev_steps):
            self.up_blocks.append(
                UpBlock(
                    in_channels=rev_steps[idx]*hidden_size * 2,
                    out_channels=rev_steps[idx+1]*hidden_size if idx < len(rev_steps) - 1 else step * hidden_size,
                    time_emb_dim=time_emb_dim,
                    is_residual=has_residual,
                    res_num=res_num,
                    attn=step in attn_step,
                    up_sample=(idx != len(steps) - 1),  # 最后一个不进行up_sample
                    is_debug=is_debug,
                )
            )

    def forward(self, x, t,skip,conds,mask):
        idx = -1
        for block in self.up_blocks:
            x = block(torch.concat([x,skip[idx]],dim=1), t,conds,mask)
            idx -= 1
        return x

class BackBone(nn.Module):
    def __init__(self, channels, time_emb_dim, is_residual=True,attn=True, is_debug=False):
        super().__init__()
        modules = []
        modules.append(
            ResnetBlock(
                in_channels=channels,
                out_channels=channels,
                time_emb_dim=time_emb_dim,
                is_residual=is_residual,
            )
        )
        if attn:
            modules.append(
                MultiheadAttention(
                    n_heads=4,
                    emb_dim=channels,
                    input_dim=channels,
                )
            )
        modules.append(
            ResnetBlock(
                in_channels=channels,
                out_channels=channels,
                time_emb_dim=time_emb_dim,
                is_residual=is_residual,
            )
        )
        self.backbone = SequenceWithTimeEmbedding(modules)

    def forward(self, x, t,conds,mask):
        return self.backbone(x, t,conds,mask)


class SequenceWithTimeEmbedding(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.models = nn.ModuleList(blocks)

    def forward(self, x, t,conds,mask):
        for model in self.models:
            x = model(x, t,conds,mask)
        return x