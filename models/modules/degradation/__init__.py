from copy import deepcopy

import torch
from torch import nn

from ..unet import DiffusionUNet
from .encoder import DegEncoder


def compute_color_map(x, norm=True):
    if norm:
        x = (x + 1) / 2
    color_map = x / (x.sum(dim=1, keepdims=True) + 1e-4)
    color_map = color_map.clip(0, 1)
    if norm:
        color_map = color_map * 2 - 1
    return color_map


class DegrationUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        config1, config2 = deepcopy(config), deepcopy(config)

        self.with_color = getattr(self.config.params, "color", True)

        num_layers = len(config.model.ch_mult)
        block_in_channels = getattr(config.model, "channels", [
            128, 128, 256, 256, 512, 512, 1024, 1024
        ][:num_layers])

        block_out_channels = list(reversed(block_in_channels))

        fix_encoder = getattr(config.params, "fixed_encoder", False)
        self.fixed_encoder = fix_encoder

        degradation = getattr(config.params, "degradation", True)
        with_deg = degradation and not self.fixed_encoder
        self.with_deg = with_deg

        channels = block_out_channels
        scales = [2] * (num_layers - 1) + [1]
        downsample = num_layers - 6

        num_deg_fusion = getattr(config.params, "num_deg_fusion", num_layers)
        channels = channels[:num_deg_fusion]
        scales = scales[:num_deg_fusion]
        block_up_channels = block_out_channels[:num_deg_fusion]

        encoder_config = getattr(config.params, "encoder", None)
        if encoder_config is None:
            encoder_config = {
                "ch": 16,
                "encoder_attn": False,
                "upsample_attn": False,
            }
        else:
            encoder_config = {
                "ch": getattr(config.params, "ch", 16),
                "encoder_attn": getattr(config.params, "encoder_attn", False),
                "upsample_attn": getattr(config.params, "upsample_attn", False),
            }

        self.encoder = DegEncoder(
            ch=encoder_config["ch"],
            channels=channels, 
            scales=scales,
            downsample=downsample,
            encoder_attn=encoder_config["encoder_attn"],
            upsample_attn=encoder_config["upsample_attn"],
        )

        if with_deg:
            config1.model.in_channels = 3
            deg_ch = getattr(config.params, "deg_ch", config.model.ch)
            config1.model.ch = deg_ch
            self.deg_unet = DiffusionUNet(
                config1,
                time_embedding=False,
                block_up_channels=block_up_channels,
            )

        self.diff_unet = DiffusionUNet(
            config2,
            time_embedding=True,
            block_up_channels=block_up_channels,
        )

        if self.fixed_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False


    def forward(self, x, t, return_latents=False):
        assert not (self.fixed_encoder and return_latents)

        if self.training:
            noisy_hr, lr, hr = x
        else:
            noisy_hr, lr = x

        if return_latents:
            features, latents1 = self.encoder(lr, return_features=True)
        else:
            features = self.encoder(lr)

        num_layers = self.diff_unet.num_layers
        encoder_features = [None] * num_layers
        decoder_features = features + [None] * (num_layers - len(features))
        features = encoder_features + decoder_features

        inputs = [noisy_hr, lr]
        if self.with_color:
            color_map = compute_color_map(lr)
            inputs.append(color_map)
        x = torch.cat(inputs, dim=1)

        pred_noise = self.diff_unet(x, t, block_inputs=features)

        if self.training and self.with_deg:
            pred_lr = self.deg_unet(hr, block_inputs=features)
            if return_latents:
                latents2 = self.encoder(pred_lr, features_only=True)
                return pred_noise, pred_lr, latents1, latents2
            return pred_noise, pred_lr
        return pred_noise

    @property
    def scale(self):
        return self.diff_unet.scale
