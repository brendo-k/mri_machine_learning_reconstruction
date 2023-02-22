import torch.nn as nn 
import torch
from .Unet_parts import down, up, concat, double_conv

class Unet(nn.Module):
    def __init__(
            self, 
            in_chan, 
            out_chan, 
            depth=4,
            chans=18, 
            with_instance_norm=False,
            drop_prob=0
            ):

        super().__init__()
        self.down_sample_layers = nn.ModuleList([double_conv(in_chan, chans, with_instance_norm, drop_prob)])
        cur_chan = chans
        for _ in range(depth):
            self.down_sample_layers.append(Unet_down(cur_chan, cur_chan*2, with_instance_norm, drop_prob))
            cur_chan *= 2

        self.up_sample_layers = nn.ModuleList()
        for _ in range(depth):
            self.up_sample_layers.append(Unet_up(cur_chan, cur_chan//2, with_instance_norm, drop_prob))
            cur_chan //= 2
        self.conv1d = nn.Conv2d(chans, out_chan, 1, bias=False)

    def forward(self, x):
        stack = []
        for layer in self.down_sample_layers:
            x = layer(x)
            stack.append(x)

        for i, layer in enumerate(self.up_sample_layers):
            x = layer(x, stack[-i - 2])
        x = self.conv1d(x)
        return x

class Unet_down(nn.Module):
    def __init__(self, in_channel, out_channel, with_instance_norm, drop_prob):
        super().__init__()
        self.down = down()
        self.conv = double_conv(in_channel, out_channel, with_instance_norm, drop_prob)

    def forward(self, x):
        x1 = self.down(x)
        x2 = self.conv(x1)
        return x2

class Unet_up(nn.Module):
    def __init__(self, in_chan, out_chan, with_instance_norm, drop_prob):
        super().__init__()
        self.concat = concat()
        self.up = up(in_chan, out_chan)
        self.conv = double_conv(in_chan, out_chan, with_instance_norm, drop_prob)

    def forward(self, x, x_encode):
        x1 = self.up(x)
        x_cat = self.concat(x_encode, x1)
        x2 = self.conv(x_cat)
        return x2

