import torch
import torch.nn as nn
from src.flow_match_dit.modules import PositionalEmbedding,TimestepEmbedder,Encoder,PatchEmbed,UnPatchify
import einops

class DiT(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_classes,
            T,
            steps=(1,2,4),
            n_heads=12,
            patch_size=4,
            hidden_size=768,
            has_residual=True,
            num_encoder= 12,
            is_debug = False
                 ):
        super(DiT, self).__init__()
        time_emb_dim = hidden_size * 4

        self.time_embedding = TimestepEmbedder(time_emb_dim)

        self.condition_embedding = nn.Sequential(
            nn.Embedding(num_classes,hidden_size),
            nn.Linear(hidden_size,time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim,time_emb_dim),
        )
        self.patch_emb = PatchEmbed(in_channels=in_channels,patch_size=patch_size,hidden_size=hidden_size)

        self.encoders = nn.ModuleList([
            Encoder(
                n_heads=n_heads,
                hidden_size=hidden_size,
                time_emb_dim=time_emb_dim,
                output_dim=hidden_size,
            ) for _ in range(num_encoder)
        ])

        self.unpatch = UnPatchify(out_channels=out_channels,patch_size=patch_size,hidden_size=hidden_size)

    def forward(self, x,t,conds,mask):
        t = self.time_embedding(t)
        conds = self.condition_embedding(conds).squeeze(dim=1)
        x,Hp,Wp = self.patch_emb(x)
        # print(x.shape)
        for encoder in self.encoders:
            x = encoder(x,t,conds,mask)
        x = self.unpatch(x,Hp,Wp)
        return x


