from monai.networks.nets.swin_unetr import SwinUNETR
import torch.nn as nn

class SwinUnet(nn.Module):
    def __init__(
            self, 
            in_chan, 
            out_chan
    ):
        self.net = SwinUNETR(img_size=(128, 128), in_channels=in_chan, out_channels=out_chan, spatial_dims=2)

    def forward(self, x):
        return self.net(x)