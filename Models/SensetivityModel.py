import torch 
import torch.nn as nn
from .Unet import Unet
from ..Utils import fft_2d_img

class SensetivityModel(nn.Module):
    def __init__(self, in_chans, out_chans, chans):
        super().__init__()
        self.model = Unet(in_chans, out_chans, chans=chans)

    # recieve coil maps as [B, C, H, W]
    def forward(self, coil_images):
        coil_images = fft_2d_img(coil_images, axes=[2, 3])
        coil_images = self.model(coil_images)
        return coil_images
