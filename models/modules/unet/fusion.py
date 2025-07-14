import torch
from torch import nn

from ..utils import ChannelAttention


class Fusion(nn.Module):
    def __init__(self, channels=None, mode="cat"):
        super().__init__()
        assert mode in ["cat", "cattn", "none"]
        self.mode = mode
        if mode == "cattn":
            self.cattns = nn.ModuleList([
                ChannelAttention(c) if c is not None else nn.Identity() for c in channels
            ])

    def forward(self, x, y=None, index=None):
        if y is None:
            return x
        if self.mode == "cat":
            return torch.cat([x, y], dim=1)
        # if self.mode == "add":
        #     return x + y
        module = self.cattns[index]        
        z = torch.cat([x, y], dim=1)
        z = module(z)
        return z
