import torch.nn as nn
import numpy as np
from ml_recon.models.NormUnet import NormUnet
# from ml_recon.models.NormUnet import NormUnet
from ml_recon.utils import ifft_2d_img
from ml_recon.utils import complex_conversion
import einops
import torch

from typing import Optional, Tuple

class SensetivityModel(nn.Module):
    def __init__(
            self, 
            in_chans: int, 
            out_chans: int, 
            chans: int, 
            mask_center:bool=True, 
            ):
        
        super().__init__()
        self.model = NormUnet(in_chans, out_chans, chans=chans)
        self.mask_center = mask_center

    # recieve coil maps as [B, C, H, W]
    def forward(self, images, mask):
        if self.mask_center:
            images = self.mask(images, mask) 

        images = ifft_2d_img(images, axes=[2, 3])
        number_of_coils = images.shape[1]
        # combine all coils along batch dimension. Leave coil dims to be 1
        images = einops.rearrange(images, 'b c h w -> (b c) 1 h w')
        # convert to real numbers 
        images = complex_conversion.complex_to_real(images)
        # pass through model
        images = self.model(images)
        # convert back to complex
        images = complex_conversion.real_to_complex(images)
        # rearange back to original format
        images = einops.rearrange(images, '(b c) 1 h w -> b c h w', c=number_of_coils)
        # rss to normalize sense maps
        images = images / self.root_sum_of_squares(images)
        return images

    def mask(self, coil_images, mask):
        squeezed_mask = mask[:, 0,  :].to(torch.int8)
        cent = squeezed_mask.shape[1] // 2
        # running argmin returns the first non-zero
        left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
        right = torch.argmin(squeezed_mask[:, cent:], dim=1)

        bounds = torch.max(torch.cat((left, right)))

        coil_images[..., :cent - bounds - 1] = 0
        coil_images[..., cent + bounds:] = 0
        return coil_images

    # sense_map [batches channel height width], take absolute root sum of squares
    def root_sum_of_squares(self, sense_map):
        sense_map = sense_map.abs().pow(2).sum(1, keepdim=True).sqrt()
        return sense_map
    