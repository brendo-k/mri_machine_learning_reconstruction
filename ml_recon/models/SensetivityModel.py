import torch.nn as nn
import numpy as np
from ml_recon.Models.Unet import Unet
from ml_recon.Utils import ifft_2d_img
from ml_recon.Utils import complex_conversion
import einops
import torch

class SensetivityModel(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, chans: int, mask_center:bool=True, acs_bounds:int=20):
        super().__init__()
        self.model = Unet(in_chans, out_chans, chans=chans, with_instance_norm=True)
        self.mask_center = mask_center
        self.acs_bounds = acs_bounds

    # recieve coil maps as [B, C, H, W]
    def forward(self, coil_images):
        if self.mask_center:
            coil_images = self.mask(coil_images) 

        coil_images = ifft_2d_img(coil_images, axes=[2, 3])
        channel_dim = coil_images.shape[1]
        # combine all coils along batch dimension. Leave coil dims to be 1
        coil_images = einops.rearrange(coil_images, 'b c h w -> (b c) 1 h w')
        # convert to real numbers 
        coil_images = complex_conversion.complex_to_real(coil_images)
        # pass through model
        sense_map = self.model(coil_images)
        # convert back to complex
        sense_map = complex_conversion.real_to_complex(sense_map)
        # rearange back to original format
        sense_map = einops.rearrange(sense_map, '(b c) 1 h w -> b c h w', c=channel_dim)
        # rss to normalize sense maps
        sense_map = (1/self.root_sum_of_squares(sense_map)) * sense_map
        return sense_map

    def mask(self, coil_images):
        image_size = coil_images.shape[3]
        center = image_size//2
        acs_bounds = [-np.ceil(self.acs_bounds/2).astype(int) + center, np.floor(self.acs_bounds/2).astype(int) + center]
        coil_images[:, :, :, :acs_bounds[0]-1] = 0
        coil_images[:, :, :, acs_bounds[1]:] = 0
        return coil_images

    # sense_map [b c h w]
    def root_sum_of_squares(self, sense_map):
        sense_map = sense_map.abs().pow(2).sum(1, keepdim=True).sqrt()
        return sense_map