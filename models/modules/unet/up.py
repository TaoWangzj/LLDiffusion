import torch
from torch import nn

from ..utils import Upsample
from .blocks import AttnBlock, ResnetBlock
from .fusion import Fusion


class UpLayers(nn.ModuleList):
    def __init__(
        self,
        block_in,
        ch,
        ch_mult,
        temb_ch,
        num_res_blocks,
        attn_resolutions,
        dropout,
        resamp_with_conv,
        block_up_channels=None,
        fuse_mode="cat",
        skip_mode="cat",
    ):
        super().__init__()
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.temb_ch = temb_ch
        scale = 1
        ups = []
        self.block_in_channels = []
        self.block_out_channels = []
        self.in_channels = block_in

        if block_up_channels is None:
            block_up_channels = [0] * self.num_layers
        n = (self.num_layers - len(block_up_channels))
        block_up_channels = block_up_channels + [0] * n

        assert skip_mode in ["cat", "add"]
        self.skip_mode = skip_mode

        ch_mult = list(reversed(ch_mult))

        for i_level in range(self.num_layers):
            blocks = nn.ModuleList()
            attn = nn.ModuleList()

            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]

            if i_level > 0:
                block_up_channel = block_up_channels[i_level - 1]
                block_in += block_up_channel

            self.block_in_channels.append(block_in + skip_in)

            if self.skip_mode == "cat":
                self.block_in_channels.append(block_in + skip_in)
            else:
                self.block_in_channels.append(block_in)

            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    if i_level < self.num_layers - 1:
                        skip_in = ch * ch_mult[i_level + 1]
                    else:
                        skip_in = ch

                if self.skip_mode == "cat":
                    block_in += skip_in
                block = ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout,
                )
                blocks.append(block)
                block_in = block_out
                if scale in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            up = nn.Module()
            up.block = blocks
            up.attn = attn
            if i_level != self.num_layers - 1:
                up.upsample = Upsample(block_in, resamp_with_conv)
                scale *= 2
            self.block_out_channels.append(block_out)
            ups.append(up)

        for up in ups:
            self.append(up)
        self.out_channels = block_in

        self.scale = scale
        assert self.scale == 2 ** (self.num_layers - 1)

        in_ch1 = [self.in_channels] + [ch * mult for mult in ch_mult][:-1]
        in_ch2 = [0] + block_up_channels[:-1]
        fuse_channels = [(c1 + c2) for c1, c2 in zip(in_ch1, in_ch2)]
        self.fuse = Fusion(fuse_channels, fuse_mode)

    def forward(self, x, skips, temb=None, block_inputs=None, return_block_outputs=False):
        if block_inputs is None:
            block_inputs = [None] * self.num_layers

        block_outputs = []
        for i_level in range(self.num_layers):

            x = self.fuse(x, block_inputs[i_level], i_level)

            for i_block in range(self.num_res_blocks + 1):

                x = self.fuse_skip(x, skips.pop())

                block = self[i_level].block[i_block]
                x = block(x, temb)

                if len(self[i_level].attn) > 0:
                    attn = self[i_level].attn[i_block]
                    x = attn(x)

            if i_level != self.num_layers - 1:
                upsample = self[i_level].upsample
                x = upsample(x)

            block_outputs.append(x)

        if return_block_outputs:
            return x, block_outputs
        return x

    def fuse_skip(self, x, y):
        if self.skip_mode == "cat":
            return torch.cat([x, y], dim=1)
        return x + y
