import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor=4, dropout=0.):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        nn.Linear(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout)
    )

class MLPMixer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, dim=512, depth=1,
                 num_classes=10, channels=3, expansion_factor=4,
                 expansion_factor_token=0.5, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0
        self.dim = dim           
        self.depth = depth  
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.token_mixing = lambda: PreNormResidual(dim,
            nn.Sequential(
                Rearrange('b n d -> b d n'),
                FeedForward(num_patches, expansion_factor, dropout),
                Rearrange('b d n -> b n d')
            )
        )

        self.channel_mixing = lambda: PreNormResidual(dim,
            FeedForward(dim, expansion_factor_token, dropout)
        )

        self.model = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
            *[nn.Sequential(
                self.token_mixing(),
                self.channel_mixing()
            ) for _ in range(depth)],
            nn.LayerNorm(dim),
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
    def get_device(self):
        return self.model[1].weight.device
    
    def reptile_update(self, other_model, num_tasks):

        with torch.no_grad():
            for p_init, p_updated in zip(self.parameters(), other_model.parameters()):
                if p_init.grad is None:
                    p_init.grad = torch.zeros_like(p_updated.data)
                p_init.grad.data.add_((p_updated.data - p_init.data)/num_tasks)