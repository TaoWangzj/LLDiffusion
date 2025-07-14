import torch
from torch import nn

from ..utils import Normalize, nonlinearity
from .down import DownLayers
from .fusion import Fusion
from .mid import MidLayer
from .time_embedding import TimeEmbeddingLayer
from .up import UpLayers


class DiffusionUNet(nn.Module):
    def __init__(self, config, time_embedding=True, block_down_channels=None, block_up_channels=None):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(
            config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.time_embedding = time_embedding
        self.temb_ch = None
        if self.time_embedding:
            self.temb_ch = self.ch * 4
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = config.model.in_channels
        self.out_channels = out_ch
        self.scale = 32

        self.block_down_channels = block_down_channels
        self.block_up_channels = block_up_channels

        # fusion
        fuse_mode = "cat"
        if hasattr(config, "params"):
            fuse_mode = getattr(config.params, "fuse_mode", "cat")
        fuse_channels = []
        if self.block_down_channels is not None:
            fuse_channels.append(self.block_down_channels[-1] * 2)
        else:
            fuse_channels.append(None)
        if self.block_up_channels is not None:
            fuse_channels.append(self.block_up_channels[-1] * 2)
        else:
            fuse_channels.append(None)
        self.fuse = Fusion(fuse_channels, fuse_mode)

        # timestep embedding
        if self.time_embedding:
            self.temb = TimeEmbeddingLayer(self.ch, self.temb_ch)

        # downsampling
        self.conv_in = nn.Conv2d(
            self.in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down = DownLayers(
            ch=ch,
            ch_mult=ch_mult,
            temb_ch=self.temb_ch,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            block_down_channels=self.block_down_channels,
            fuse_mode="none" if self.block_down_channels is None else fuse_mode,
        )

        # middle
        block_in = self.down.out_channels
        block_out = block_in
        if self.block_down_channels is not None:
            block_in += self.block_down_channels[-1]

        self.mid = MidLayer(block_in, block_out, self.temb_ch, dropout)

        block_in = block_out

        # upsampling
        skip_mode = getattr(config.model, "skip_mode", "cat")
        self.up = UpLayers(
            block_in=block_in,
            ch=ch,
            ch_mult=ch_mult,
            temb_ch=self.temb_ch,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            block_up_channels=self.block_up_channels,
            fuse_mode="none" if self.block_up_channels is None else fuse_mode,
            skip_mode=skip_mode,
        )

        block_in = self.up.out_channels
        if self.block_up_channels is not None:
            if len(self.block_up_channels) == self.num_layers:
                block_in += self.block_up_channels[-1]

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, t=None, return_blocks=False, block_inputs=None):
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x, dim=1)

        h, w = x.shape[-2:]
        assert h % self.scale == 0 and w % self.scale == 0

        n = self.num_layers
        if block_inputs is None:
            block_inputs = [None] * (2 * n)

        down_injections = [None] + block_inputs[:n - 1]
        mid_injection = block_inputs[n - 1]
        up_injections = [None] + block_inputs[n:-1]
        final_injection = block_inputs[-1]

        # timestep embedding
        if self.time_embedding:
            t = self.temb(t)

        # downsampling
        h = self.conv_in(x)

        skips = self.down(h, temb=t, block_inputs=down_injections,
                          return_block_outputs=return_blocks)
        if return_blocks:
            skips, block_down_outputs = skips

        # middle
        h = skips[-1]

        h = self.fuse(h, mid_injection, 0)

        h = self.mid(h, t)

        # upsampling
        h = self.up(h, skips, t, block_inputs=up_injections,
                    return_block_outputs=return_blocks)
        if return_blocks:
            h, block_up_outputs = h

        h = self.fuse(h, final_injection, 1)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        if return_blocks:
            block_outputs = block_down_outputs + block_up_outputs
            return h, block_outputs
        return h
