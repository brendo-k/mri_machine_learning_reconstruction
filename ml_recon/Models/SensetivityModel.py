import torch.nn as nn
import numpy as np
from .Unet import Unet
from Utils import fft_2d_img
from Utils import complex_conversion
import einops
import torch

class SensetivityModel(nn.Module):
    def __init__(self, in_chans, out_chans, chans, mask_center=True, acs_bounds=30):
        super().__init__()
        self.model = Unet(in_chans, out_chans, chans=chans, with_instance_norm=True)
        self.mask_center = mask_center
        self.acs_bounds = acs_bounds

    # recieve coil maps as [B, C, H, W]
    def forward(self, coil_images):
        if self.mask_center:
            coil_images = self.mask(coil_images) 

        coil_images = fft_2d_img(coil_images, axes=[2, 3])
        coil_images = einops.rearrange(coil_images, 'b c h w -> (b c) h w')
        coil_images = coil_images[:, None, :, :]
        coil_images_real = complex_conversion.complex_to_real(coil_images)
        sense_map = self.model(coil_images_real)
        sense_map = complex_conversion.real_to_complex(sense_map)
        sense_map = sense_map.permute((1, 0, 2, 3))
        sense_map = (1/self.root_sum_of_squares(sense_map)) * sense_map
        return sense_map

    def mask(self, coil_images):
        image_size = coil_images.shape[3]
        center = image_size//2
        acs_bounds = [-np.ceil(self.acs_bounds/2).astype(int) + center, np.floor(self.acs_bounds/2).astype(int) + center]
        coil_images[:, :, :, :acs_bounds[0]] = 0
        coil_images[:, :, :, acs_bounds[1]:] = 0
        return coil_images

    # sense_map [b c h w]
    def root_sum_of_squares(self, sense_map):
        sense_map = torch.sqrt(torch.sum(sense_map.abs() ** 2, dim=1, keepdim=True))
        return sense_map