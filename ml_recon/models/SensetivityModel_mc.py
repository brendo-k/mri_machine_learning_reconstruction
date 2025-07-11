import einops
import torch
import math

import torch.nn as nn
from ml_recon.utils import ifft_2d_img
from ml_recon.models import Unet
from ml_recon.utils import real_to_complex, complex_to_real, root_sum_of_squares

from typing import Tuple, Literal

class SensetivityModel_mc(nn.Module):
    def __init__(
            self, 
            in_chans: int, 
            out_chans: int, 
            chans: int, 
            conv_after_upsample: bool = False,
            upsample_method: Literal['conv', 'max', 'bilinear'] = 'conv'
            ):

        """Module used to estimate sensetivity maps based on masked k-space center

        Args:
            in_chans (int): _description_
            out_chans (int): _description_
            chans (int): Number of convolutional channels for U-Net estimator
            conv_after_upsample (bool): boolean flag for addition of 3x3 convolution after upsampling
            upsample_method (str): upsampling method for U-Net ('bilinear', 'max', 'conv')
        """
        super().__init__()
        self.model = Unet(
            in_chans, 
            out_chans, 
            chans=chans, 
            conv_after_upsample=conv_after_upsample, 
            upsample_method=upsample_method
            )

    # recieve coil maps as [B, contrast, channels, H, W]
    def forward(self, images, mask):
        images = self.mask_center(images, mask) 
        # get the first image for estimating coil sensetivites
        images = images[:, [0], :, :, :]

        images = ifft_2d_img(images, axes=[-1, -2])
        assert isinstance(images, torch.Tensor)

        number_of_coils = images.shape[2]
        num_contrasts = images.shape[1]

        images = einops.rearrange(images, 'b contrast c h w -> (b contrast c) 1 h w')
        assert isinstance(images, torch.Tensor)

        # convert to real numbers [b * contrast * coils, cmplx, h, w]
        images = complex_to_real(images)
        # norm 
        images, mean, std = self.norm(images)
        assert not torch.isnan(images).any()
        # pass through model
        images = self.model(images)
        assert not torch.isnan(images).any()
        # unnorm
        images = self.unnorm(images, mean, std)
        # convert back to complex
        images = real_to_complex(images)
        # rearange back to original format
        images = einops.rearrange(images, '(b contrast c) 1 h w -> b contrast c h w', c=number_of_coils, contrast=num_contrasts)
        # rss to normalize sense maps
        rss_norm = root_sum_of_squares(images, coil_dim=2).unsqueeze(2) 
        images = images / rss_norm
        return images

    def mask_center(self, coil_k_spaces, mask):
        # I did some strange things here. Before, I tried to find the largest 2d box 
        # that was continuously contained in sampled k-space. However, that added some extra 
        # bugs in self-suprvised training as the center box size could change depending
        # on the sets. I have just hard coded this for now, but could be interesting to 
        # test different coil estimation methods and masking methods. 
        # 
        # There are probably some interesting ideas for deep learning coil estimatation methods. 
        h, w = coil_k_spaces.shape[-2:] 
        center_mask = torch.zeros((h, w), device=coil_k_spaces.device)
        center_mask[h//2-5:h//2+5, w//2-5:w//2+5] = 1

        return coil_k_spaces * center_mask
        

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # instance norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean
