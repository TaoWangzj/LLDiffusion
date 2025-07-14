import math
from functools import lru_cache

import torch
from torch import nn

from ..utils import nonlinearity


@lru_cache()
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimeEmbeddingLayer(nn.Module):
    def __init__(self, ch, temb_ch):
        super().__init__()
        self.ch = ch
        self.temb_ch = temb_ch
        self.dense = nn.ModuleList([
            nn.Linear(self.ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

    def forward(self, t):
        temb = get_timestep_embedding(t, self.ch)
        temb = self.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.dense[1](temb)
        return temb
