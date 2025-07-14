"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from itertools import chain

import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from ..unet.blocks import AttnBlock


def get_nonspade_norm_layer(norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            print(
                'SyncBatchNorm is currently not supported under single-GPU mode, switch to "instance" instead')
            # norm_layer = SynchronizedBatchNorm2d(
            #    get_out_channel(layer), affine=True)
            norm_layer = nn.InstanceNorm2d(
                get_out_channel(layer), affine=False)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(
                get_out_channel(layer), affine=False)
        else:
            raise ValueError(
                f'normalization layer {subnorm_type} is not recognized')

        return nn.Sequential(layer, norm_layer)

    # print('This is a legacy from nvlabs/SPADE, and will be removed in future versions.')
    return add_norm_layer


class ResnetBlock(nn.Module):
    def __init__(self, dim, act, kernel_size=3):
        super().__init__()
        self.act = act
        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            act,
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size)
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return self.act(out)


class ConvEncoder(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, ch=16, attn=False):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        self.ch = ch
        self.attn = attn
        norm_E = "spectralinstance"
        norm_layer = get_nonspade_norm_layer(norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ch, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(
            nn.Conv2d(ch * 1, ch * 2, kw, stride=2, padding=pw))

        self.layer3 = norm_layer(
            nn.Conv2d(ch * 2, ch * 4, kw, stride=2, padding=pw))

        self.layer4 = norm_layer(
            nn.Conv2d(ch * 4, ch * 8, kw, stride=2, padding=pw))

        self.layer5 = norm_layer(
            nn.Conv2d(ch * 8, ch * 16, kw, stride=2, padding=pw))

        self.res_0 = ResnetBlock(ch*16, nn.LeakyReLU(0.2, False))

        self.res_1 = ResnetBlock(ch*16, nn.LeakyReLU(0.2, False))
        self.res_2 = ResnetBlock(ch*16, nn.LeakyReLU(0.2, False))

        if self.attn:
            self.attn_0 = AttnBlock(ch * 2)
            self.attn_1 = AttnBlock(ch * 8)
            self.attn_2 = AttnBlock(ch * 16)
            self.attn_3 = AttnBlock(ch * 16)

        activation = nn.LeakyReLU(0.2, False)

        self.out = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(ch * 16, ch * 32, kernel_size=kw)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(ch * 32, ch * 32, kernel_size=kw)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(ch * 32, ch * 32, kernel_size=kw)),
            activation
        )
        self.down = nn.AvgPool2d(2, 2)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.pad_3 = nn.ReflectionPad2d(3)
        self.pad_1 = nn.ReflectionPad2d(1)
        self.conv_7x7 = nn.Conv2d(
            ch, ch, kernel_size=7, padding=0, bias=True)
        # self.opt = opt

    def forward(self, x, return_features=False):
        # if x.size(2) != 256 or x.size(3) != 256:
        #     x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = self.layer1(x)  # 128
        x = self.conv_7x7(self.pad_3(self.actvn(x)))
        x = self.layer2(self.actvn(x))  # 64
        if self.attn:
            x = self.attn_0(x)
        x1 = x
        x = self.layer3(self.actvn(x))  # 32
        x = self.layer4(self.actvn(x))  # 16
        if self.attn:
            x = self.attn_1(x)
        x2 = x
        x = self.layer5(self.actvn(x))  # 8

        x = self.res_0(x)
        if self.attn:
            x = self.attn_2(x)
        x3 = x
        x = self.res_1(x)
        x = self.res_2(x)
        if self.attn:
            x = self.attn_3(x)
        x4 = x

        mu = self.out(x)

        if return_features:
            return mu, [x1, x2, x3, x4]
        return mu


class ConvEncoderLoss(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        self.ndf = ndf
        norm_E = "spectralinstance"
        norm_layer = get_nonspade_norm_layer(None, norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(
            nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        # self.layer2_1 = norm_layer(nn.Conv2d(ndf * 2, ndf * 2, kw, stride=1, padding=pw))
        self.layer3 = norm_layer(
            nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        # self.layer3_1 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, kw, stride=1, padding=pw))
        self.layer4 = norm_layer(
            nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        # self.layer4_1 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))
        self.layer5 = norm_layer(
            nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(
            nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))
        # self.layer7 = norm_layer(nn.Conv2d(ndf * 2, ndf * 2, kw, stride=1, padding=pw))
        # self.layer8 = norm_layer(nn.Conv2d(ndf * 2, ndf * 2, kw, stride=1, padding=pw))
        self.so = s0 = 4
        self.out = norm_layer(
            nn.Conv2d(ndf * 8, ndf * 4, kw, stride=1, padding=0))
        self.down = nn.AvgPool2d(2, 2)
        # self.global_avg = nn.AdaptiveAvgPool2d((6,6))

        self.actvn = nn.LeakyReLU(0.2, False)
        self.pad_3 = nn.ReflectionPad2d(3)
        self.pad_1 = nn.ReflectionPad2d(1)
        self.conv_7x7 = nn.Conv2d(
            ndf, ndf, kernel_size=7, padding=0, bias=True)
        # self.opt = opt

    def forward(self, x):

        x1 = self.layer1(x)  # 128
        x2 = self.conv_7x7(self.pad_3(self.actvn(x1)))
        x3 = self.layer2(self.actvn(x2))  # 64
        # x = self.layer2_1(self.actvn(x))
        x4 = self.layer3(self.actvn(x3))  # 32
        # x = self.layer3_1(self.actvn(x))
        x5 = self.layer4(self.actvn(x4))  # 16
        # x = self.layer4_1(self.actvn(x))
        return [x1, x2, x3, x4, x5]


class EncodeMap(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(
            nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(
            nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(
            nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(
            nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(
                nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))

        s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.layer_final = nn.Conv2d(
            ndf * 8, ndf * 16, kw, stride=1, padding=pw)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        # if self.opt.crop_size >= 256:
        #     x = self.layer6(self.actvn(x))
        x = self.actvn(x)
        return self.layer_final(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar


class ResizeLayer(nn.Sequential):
    def __init__(self, scale=2, channels=None, param=True):
        super().__init__()
        assert scale in [0.5, 1, 2]
        module = nn.Identity()
        if scale == 0.5:
            if param:
                module = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
            else:
                module = nn.AvgPool2d(2, 2)
        elif scale == 2:
            if param:
                module = nn.ConvTranspose2d(
                    channels, channels, 3, stride=2, padding=1, output_padding=1)
            else:
                module = nn.Upsample(scale_factor=scale)
        self.module = module


class ResizeConvBlock(nn.Sequential):
    def __init__(self, dim_in, dim_out, scale=2, kernel_size=3, param=True, attn=False):
        pw = (kernel_size - 1) // 2
        layers = [
            ResizeLayer(scale, dim_in, param),
            nn.ReflectionPad2d(pw),
        ]
        if attn:
            layers.append(AttnBlock(dim_in))
        layers.append(spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # nn.ReflectionPad2d(pw)
        # spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size))
        # nn.LeakyReLU(0.2)
        super().__init__(*layers)



class PriorUpsample(nn.Module):
    def __init__(
        self,
        channels=(512, 512, 256, 256, 128, 128),
        scales=(2, 2, 2, 2, 2, 1),
        downsample=0,
        attn=False,
    ):
        assert len(channels) == len(scales)
        super().__init__()
        adjacent_channels = zip(chain([512], channels[:-1]), channels)
        if downsample != 0:
            convs = []
            for _ in range(downsample):
                convs.append(ResizeConvBlock(512, 512, 0.5, attn=attn))
            self.down = nn.Sequential(*convs)
        self.convs = nn.ModuleList([
            ResizeConvBlock(c1, c2, scale, attn=attn)
            for (c1, c2), scale in zip(adjacent_channels, scales)
        ])

    def forward(self, x):
        xs = []
        if hasattr(self, "down"):
            x = self.down(x)
        for conv in self.convs:
            x = conv(x)
            xs.append(x)
        return xs


class DegEncoder(nn.Module):
    def __init__(
        self,
        ch=16,
        channels=(512, 512, 512, 256, 256, 128, 128),
        scales=(2, 2, 2, 2, 2, 1),
        downsample=0,
        encoder_attn=False,
        upsample_attn=False,
    ):
        super().__init__()
        self.encoder = ConvEncoder(ch=ch, attn=encoder_attn)
        self.upsample = PriorUpsample(
            channels=channels,
            scales=scales,
            downsample=downsample,
            attn=upsample_attn,
        )

    def forward(self, x, return_features=False, features_only=False):
        assert not (return_features and features_only)
        if return_features or features_only:
            x, features = self.encoder(x, return_features=True)
        else:
            x = self.encoder(x)
        if features_only:
            return features
        x = self.upsample(x)
        if return_features:
            return x, features
        return x


if __name__ == "__main__":
    device = "cpu"
    import torch
    encoder = ConvEncoder().to(device)
    image_size = 64
    image = torch.randn(1, 3, image_size, image_size).to(device)
    output = encoder(image)
    print(output.shape)
    upsample = PriorUpsample().to(device)
    outputs = upsample(output)
    for output in outputs:
        print(output.shape)
