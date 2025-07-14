import torch
from torch import nn

from ..unet import DiffusionUNet
from ..utils import Normalize


class MultiheadUNet(DiffusionUNet):
    def __init__(self, config, num_heads=2):
        super().__init__(config, time_embedding=False)
        assert num_heads == 2
        self.num_heads = num_heads
        self.pos = -2
        in_channels = self.up.block_out_channels[self.pos]
        out_channels = config.model.out_ch
        self.head = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.SiLU(inplace=True),
            # Normalize(in_channels),
            # nn.ConvTranspose2d(in_channels, in_channels // 2, 4, stride=2, padding=1),
            # nn.SiLU(inplace=True),
            Normalize(in_channels // 2),
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.Sigmoid(),
        ])

    def forward(self, *args, return_blocks=False, **kwargs):
        h, block_outputs = super().forward(*args, **kwargs, return_blocks=True)
        head_input = block_outputs[self.pos]
        head_output = self.head(head_input)
        if return_blocks:
            return h, head_output, block_outputs
        return h, head_output


class TwinUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        from copy import deepcopy
        config1, config2 = deepcopy(config), deepcopy(config)
        config1.model.in_channels = 6
        self.unet1 = MultiheadUNet(config1)

        block_in_channels = [128, 128, 256, 256, 512, 512]
        block_out_channels = list(reversed(block_in_channels))
        self.unet2 = DiffusionUNet(
            config2,
            block_down_channels=block_in_channels,
            block_up_channels=block_out_channels,
        )

    def forward(self, x, t):
        noisy_x, cond_x = x.chunk(2, dim=1)
        x = torch.cat([noisy_x, cond_x], dim=1)
        pred_image, pred_color, block_outputs = self.unet1(
            x, return_blocks=True)

        # (0, 1) -> (-1, 1)
        norm_pred_color = pred_color * 2 - 1
        x = torch.cat([noisy_x, cond_x, norm_pred_color], dim=1)
        pred_noise = self.unet2(x, t, block_inputs=block_outputs)

        if self.training:
            return pred_noise, pred_image, pred_color
        # TODO
        # pred_image and color will not be changed when testing
        # so we can cache them
        return pred_noise
