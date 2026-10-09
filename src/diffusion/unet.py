import torch
import torch.nn as nn
from src.diffusion.modules import PositionalEmbedding,DownBlocks,UpBlocks,BackBone

class UNet(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            T,
            steps=(1,2,4),
            hidden_size=128,
            attn_step= (1,),
            has_residual=True,
            num_resolution_blocks=2,
            is_debug = False
                 ):
        super(UNet, self).__init__()
        time_emb_dim = hidden_size * 4
        self.time_embedding = nn.Sequential(
            PositionalEmbedding(T,hidden_size),
            nn.Linear(hidden_size,time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim,time_emb_dim),
        )
        self.first_conv =nn.Conv2d(in_channels, steps[0]*hidden_size, kernel_size=3, padding=1)
        self.down_blocks = DownBlocks(
            time_emb_dim,
            steps=steps,
            hidden_size=hidden_size,
            attn_step= list(attn_step),
            res_num=num_resolution_blocks,
            has_residual=has_residual,
            is_debug=is_debug
        )
        self.backbone =BackBone(
            channels=steps[-1] * hidden_size,
            time_emb_dim=time_emb_dim,
            is_debug=is_debug,
            is_residual=has_residual,
            attn=True
        )
        self.up_blocks = UpBlocks(
            time_emb_dim,
            steps=steps,
            hidden_size=hidden_size,
            attn_step=list(attn_step),
            res_num=num_resolution_blocks,
            has_residual=has_residual,
            is_debug=is_debug
        )

        self.out = nn.Sequential(*[
            nn.GroupNorm(8, steps[0] * hidden_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=steps[0] * hidden_size, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        ])

    def forward(self, x,t):
        t = self.time_embedding(t)
        x = self.first_conv(x)
        # print(x.shape)
        x,skip = self.down_blocks(x,t)
        x = self.backbone(x,t)
        x = self.up_blocks(x,t,skip)
        x = self.out(x)
        return x


