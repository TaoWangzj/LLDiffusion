from ..utils import TimeEmbeddingSequential
from .blocks import AttnBlock, ResnetBlock


class MidLayer(TimeEmbeddingSequential):
    def __init__(self, block_in, block_out, temb_ch, dropout):
        super().__init__()
        self.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=temb_ch,
            dropout=dropout,
        )
        self.attn_1 = AttnBlock(block_in)
        self.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_out,
            temb_channels=temb_ch,
            dropout=dropout,
        )
