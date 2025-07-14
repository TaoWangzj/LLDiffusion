import torch
from torch import nn

from .utils import Normalize, nonlinearity


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels,
                                       out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                               out_channels,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                              out_channels,
                                              1)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, head_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.head_size = head_size
        self.att_size = in_channels // head_size
        assert (in_channels % head_size ==
                0), 'The size of head should be divided by the number of channels.'
        n = in_channels
        self.norm1 = Normalize(n)
        self.norm2 = Normalize(n)

        self.q = nn.Conv2d(n, n, 1)
        self.k = nn.Conv2d(n, n, 1)
        self.v = nn.Conv2d(n, n, 1)
        self.proj_out = nn.Conv2d(n, n, 1)
        self.num = 0

    def forward(self, x, y=None):
        h_ = x
        h_ = self.norm1(h_)
        if y is None:
            y = h_
        else:
            y = self.norm2(y)

        q = self.q(y)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, self.head_size, self.att_size, h*w)
        q = q.permute(0, 3, 1, 2)  # b, hw, head, att

        k = k.reshape(b, self.head_size, self.att_size, h*w)
        k = k.permute(0, 3, 1, 2)

        v = v.reshape(b, self.head_size, self.att_size, h*w)
        v = v.permute(0, 3, 1, 2)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        scale = int(self.att_size) ** (-0.5)
        q.mul_(scale)
        w_ = torch.matmul(q, k)
        w_ = w_.softmax(dim=3)

        w_ = w_.matmul(v)

        w_ = w_.transpose(1, 2).contiguous()  # [b, h*w, head, att]
        w_ = w_.view(b, h, w, -1)
        w_ = w_.permute(0, 3, 1, 2)

        w_ = self.proj_out(w_)

        return x + w_
