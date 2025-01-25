import math
from typing import Tuple, List

import torch.nn as nn 
import torch
import torch.nn.functional as F
from argparse import ArgumentParser

class Unet(nn.Module):


    def __init__(
            self, 
            in_chan:int, 
            out_chan:int, 
            depth:int=4,
            chans:int=18, 
            drop_prob:float=0.0, 
            relu_slope:float=0.2
            ):

        super().__init__()
        self.down_sample_layers = nn.ModuleList([double_conv(in_chan, chans, drop_prob, relu_slope)])
        cur_chan = chans
        for _ in range(depth):
            self.down_sample_layers.append(Unet_down(cur_chan, cur_chan*2, drop_prob, relu_slope))
            cur_chan *= 2

        self.up_sample_layers = nn.ModuleList()
        for _ in range(depth):
            self.up_sample_layers.append(Unet_up(cur_chan, cur_chan//2, drop_prob, relu_slope))
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
        w_mult = w + (16 - w % 16) # move to next multiple of 16
        h_mult = h + (16 - h % 16) # h that is the next multiple of 16
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
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
    def __init__(self, in_channel, out_channel, drop_prob, relu_slope):
        super().__init__()
        self.down = down()
        self.conv = double_conv(in_channel, out_channel, drop_prob, relu_slope)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x

class Unet_up(nn.Module):
    def __init__(self, in_chan, out_chan, drop_prob, relu_slope):
        super().__init__()
        self.up = up(in_chan, out_chan)
        self.concat = concat()
        self.conv = double_conv(in_chan, out_chan, drop_prob, relu_slope)

    def forward(self, x, x_concat):
        x = self.up(x)
        x = self.concat(x, x_concat)
        x = self.conv(x)
        return x


class double_conv(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob, relu_slope):
        
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans, affine=True),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans, affine=True),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.Dropout2d(drop_prob),
        )
      
    def forward(self, x):
        return self.layers(x)

class down(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.AvgPool2d(2, stride=(2, 2))

    def forward(self, x):
        x = self.max_pool(x)
        return x


class up(nn.Module):

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.layers = nn.Sequential(
          nn.ConvTranspose2d(in_chan, out_chan, stride=2, kernel_size=2, bias=False),
          nn.InstanceNorm2d(out_chan, affine=True),
          nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, x_concat: torch.Tensor):
        x_concat_shape = x_concat.shape[-2:]
        x_shape = x.shape[-2:]
        diff_x = x_concat_shape[0] - x_shape[0]
        diff_y = x_concat_shape[1] - x_shape[1]
        x_concat_trimmed = x_concat
        if diff_x != 0:
            x_concat_trimmed = x_concat_trimmed[:, :, diff_x//2:-diff_x//2, :]
        if diff_y != 0:
            x_concat_trimmed = x_concat_trimmed[:, :, :, diff_y//2:-diff_y//2]
        concated_data = torch.cat((x, x_concat_trimmed), dim=1)
        return concated_data