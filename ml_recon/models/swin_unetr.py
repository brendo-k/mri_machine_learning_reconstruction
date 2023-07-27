from monai.networks.nets.swin_unetr import SwinUNETR
import torch.nn as nn

class SwinUnet(nn.Module):
    def __init__(
            self, 
            in_chan, 
            out_chan, 
            image_size=(128, 128)
    ):
        self.net = SwinUNETR(img_size=image_size, in_channels=in_chan, out_channels=out_chan)

    def forward(self, x):
        return self.net(x)