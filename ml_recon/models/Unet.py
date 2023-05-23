import math
from typing import Tuple, List

import torch.nn as nn 
import torch
import torch.nn.functional as F
from .Unet_parts import down, up, concat, double_conv

class Unet(nn.Module):
    def __init__(
            self, 
            in_chan, 
            out_chan, 
            depth=4,
            chans=18, 
            drop_prob=0
            ):

        super().__init__()
        self.down_sample_layers = nn.ModuleList([double_conv(in_chan, chans, drop_prob)])
        cur_chan = chans
        for _ in range(depth):
            self.down_sample_layers.append(Unet_down(cur_chan, cur_chan*2, drop_prob))
            cur_chan *= 2

        self.up_sample_layers = nn.ModuleList()
        for _ in range(depth):
            self.up_sample_layers.append(Unet_up(cur_chan, cur_chan//2, drop_prob))
            cur_chan //= 2
        self.conv1d = nn.Conv2d(chans, out_chan, 1, bias=False)

    def forward(self, x):
        x, pad_sizes = self.pad(x)
        stack = []
        for layer in self.down_sample_layers:
            x = layer(x)
            stack.append(x)

        for i, layer in enumerate(self.up_sample_layers):
            x = layer(x, stack[-i - 2])
        x = self.conv1d(x)
        x = self.unpad(x, *pad_sizes)
        return x

    # pad input image to be divisible by 16 for unet downsampling
    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    # unpad unet input
    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]



class Unet_down(nn.Module):
    def __init__(self, in_channel, out_channel, drop_prob):
        super().__init__()
        self.down = down()
        self.conv = double_conv(in_channel, out_channel, drop_prob)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x

class Unet_up(nn.Module):
    def __init__(self, in_chan, out_chan, drop_prob):
        super().__init__()
        self.concat = concat()
        self.up = up(in_chan, out_chan)
        self.conv = double_conv(in_chan, out_chan, drop_prob)

    def forward(self, x, x_encode):
        x = self.up(x)
        x = self.concat(x_encode, x)
        x = self.conv(x)
        return x