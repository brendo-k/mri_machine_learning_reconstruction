from torch import nn
from ml_recon.models.UNet import concat, double_conv, Unet_down

class x_net(nn.Module):
    def __init__(self, contrast_order, channels, depth):
        super().__init__()
        self.contrast_order = contrast_order
        self.channels = channels
        self.depth = depth

        n_contrasts = len(self.contrast_order)
        
        input_channels = n_contrasts * 2
        channels_per_contrast = channels * n_contrasts
        self.down_sample_layers = nn.ModuleList([
            double_conv(input_channels, channels_per_contrast, drop_prob=0, groups=n_contrasts)
            ])
        cur_chan = channels
        for _ in range(depth):
            self.down_sample_layers.append(Unet_down(cur_chan, cur_chan*2*n_contrasts, drop_prob=0, groups=n_contrasts))
            cur_chan *= 2

