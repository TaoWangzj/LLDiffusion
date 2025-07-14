from torch import nn

from .decoder import MultiHeadDecoder
from .encoder import MultiHeadEncoder
from .vector_quantizer import VectorQuantizer


class VQVAEGAN(nn.Module):
    def __init__(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, temb_channels=0, ch_mult=(1, 2, 4, 8),
                 num_res_blocks=2, attn_resolutions=(16,), dropout=0.0, in_channels=3,
                 resolution=512, z_channels=256, double_z=False, enable_mid=True,
                 fix_decoder=False, fix_codebook=False, head_size=1, **ignore_kwargs):
        super().__init__()

        self.encoder = MultiHeadEncoder(
            ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
            resolution=resolution, z_channels=z_channels, double_z=double_z,
            enable_mid=enable_mid, head_size=head_size, temb_channels=temb_channels,
        )
        self.decoder = MultiHeadDecoder(
            ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
            resolution=resolution, z_channels=z_channels, enable_mid=enable_mid,
            head_size=head_size, temb_channels=temb_channels,
        )

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        self.quant_conv = nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)

        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False

    def encode(self, x, temb=None):

        hs = self.encoder(x, temb=temb)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, hs

    def decode(self, quant, hs, temb=None):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, temb=temb)

        return dec

    def forward(self, input, temb=None):
        quant, diff, info, hs = self.encode(input, temb=temb)
        dec = self.decode(quant, hs, temb=temb)

        return dec, diff, info, hs
