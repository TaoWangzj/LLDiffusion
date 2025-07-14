from .vq_transformer import VQVAEGANMultiHeadTransformer

from ..unet.time_embedding import TimeEmbeddingLayer


class RestoreFormer(VQVAEGANMultiHeadTransformer):
    def __init__(self, in_channels=3, out_channels=3, ch=128, temb_channels=0, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_ch=out_channels,
            ch=ch,
            temb_channels=temb_channels,
            **kwargs,
        )

        if temb_channels > 0:
            self.temb = TimeEmbeddingLayer(ch, temb_ch=temb_channels)

    def forward(self, x, t=None):
        if t is not None:
            t = self.temb(t)

        dec, diff, info, hs = super().forward(x, temb=t)
        return dec, diff
