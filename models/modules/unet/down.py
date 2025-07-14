from torch import nn

from ..utils import Downsample
from .blocks import AttnBlock, ResnetBlock
from .fusion import Fusion


class DownLayers(nn.ModuleList):
    def __init__(
        self,
        ch,
        ch_mult,
        temb_ch,
        num_res_blocks,
        attn_resolutions,
        dropout,
        resamp_with_conv,
        block_down_channels=None,
        fuse_mode="cat",
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.temb_ch = temb_ch
        self.num_layers = len(ch_mult)
        self.block_in_channels = []
        self.block_out_channels = []

        if block_down_channels is None:
            block_down_channels = [0] * self.num_layers
        n = (self.num_layers - len(block_down_channels))
        block_down_channels = block_down_channels + [0] * n

        self.in_channels = ch
        scale = 2 ** (self.num_layers - 1)
        self.scale = scale
        for i_level in range(self.num_layers):
            blocks = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * ((1,) + ch_mult)[i_level]
            block_out = ch * ch_mult[i_level]
            if block_down_channels is not None:
                if i_level > 0:
                    block_in += block_down_channels[i_level - 1]
            self.block_in_channels.append(block_in)
            for i_block in range(self.num_res_blocks):
                blocks.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout,
                ))
                block_in = block_out
                if scale in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            down = nn.Module()
            down.block = blocks
            down.attn = attn
            if i_level != self.num_layers - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                scale //= 2
            self.block_out_channels.append(block_out)
            self.append(down)

        assert scale == 1
        self.out_channels = block_in

        in_ch1 = [self.in_channels] + [ch * mult for mult in ch_mult][:-1]
        in_ch2 = [0] + block_down_channels[:-1]
        fuse_channels = [(c1 + c2) for c1, c2 in zip(in_ch1, in_ch2)]
        self.fuse = Fusion(fuse_channels, fuse_mode)

    def forward(self, h, temb, block_inputs=None, return_block_outputs=False):
        block_outputs = []
        hs = [h]
        if block_inputs is None:
            block_inputs = [None] * self.num_layers
        for i_level in range(self.num_layers):
            for i_block in range(self.num_res_blocks):
                block = self[i_level].block[i_block]
                h = hs[-1]
                if i_block == 0:
                    h = self.fuse(h, block_inputs[i_level])
                h = block(h, temb)
                if len(self[i_level].attn) > 0:
                    attn = self[i_level].attn[i_block]
                    h = attn(h)
                hs.append(h)
            if i_level != self.num_layers - 1:
                h = self[i_level].downsample(hs[-1])
                hs.append(h)
            block_outputs.append(h)
        if return_block_outputs:
            return hs, block_outputs
        return hs
