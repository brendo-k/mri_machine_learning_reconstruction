import einops
import torch
import math

import torch.nn as nn
from ml_recon.utils import ifft_2d_img
from ml_recon.models import Unet
from ml_recon.utils import real_to_complex, complex_to_real, root_sum_of_squares

from typing import Tuple

class SensetivityModel_mc(nn.Module):
    def __init__(
            self, 
            in_chans: int, 
            out_chans: int, 
            chans: int, 
            mask_center: bool = True, 
            batch_contrasts: bool = False,
            ):
        
        super().__init__()
        self.batch_contrasts = batch_contrasts
        self.model = Unet(in_chans, out_chans, chans=chans)
        self.mask_center = mask_center

    # recieve coil maps as [B, contrast, channels, H, W]
    def forward(self, images, mask):
        assert not torch.isnan(images).any()
        if self.mask_center:
            images = self.mask(images, mask) 
        assert not torch.isnan(images).any()
        # get the first image for estimating coil sensetivites
        images = images[:, [0], :, :, :]

        images = ifft_2d_img(images, axes=[-1, -2])

        number_of_coils = images.shape[2]
        num_contrasts = images.shape[1]

        if self.batch_contrasts:
            images = einops.rearrange(images, 'b contrast c h w -> (b c) contrast h w')
        else:
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
        if self.batch_contrasts:
            images = einops.rearrange(images, '(b c) contrast h w -> b contrast c h w', c=number_of_coils, contrast=num_contrasts)
        else:
            images = einops.rearrange(images, '(b contrast c) 1 h w -> b contrast c h w', c=number_of_coils, contrast=num_contrasts)
        # rss to normalize sense maps
        rss_norm = root_sum_of_squares(images, coil_dim=2).unsqueeze(2) + 1e-9
        #assert not (rss_norm == 0).any()
        images = images / rss_norm
        return images

    def mask(self, coil_k_spaces, center_mask):
        # coil_k: [b cont chan height width]
        masked_k_space = coil_k_spaces.clone()
        center_x = center_mask.shape[-1] // 2
        center_y = center_mask.shape[-2] // 2
        
        # height doesn't matter (since column wise sampling) 
        squeezed_mask = (center_mask[:, :, 0, center_y, :] > 0.75).to(torch.int8)
        squeezed_mask = (center_mask[:, :, 0, :, center_x] > 0.75).to(torch.int8)
        # Get the first zero index starting from the center. This gives us "left"
        # and "right" sides of ACS
        left = torch.argmin(squeezed_mask[..., :center_x].flip(-1), dim=-1)
        right = torch.argmin(squeezed_mask[..., center_x:], dim=-1)
        top = torch.argmin(squeezed_mask[..., :center_y].flip(-1), dim=-1)
        bottom = torch.argmin(squeezed_mask[..., center_y:], dim=-1)

        assert (left != 0).any(), 'Left mask bounds should be more than 1!'
        assert (right != 0).any(), 'Right mask bounds should be more than 1!'
        assert (top != 0).any(), 'Left mask bounds should be more than 1!'
        assert (bottom != 0).any(), 'Right mask bounds should be more than 1!'

        # force symmetric left and right acs boundries
        low_freq_x = torch.min(left, right)
        low_freq_y = torch.min(top, bottom)

        center_mask = torch.zeros_like(masked_k_space, dtype=torch.bool)
        # loop through num_low freq tensor and set acs lines to true
        for i in range(low_freq_x.shape[0]):
            for j in range(low_freq_y.shape[1]):
                center_mask[i, j, :, center_y - low_freq_y[i, j]:center_y + low_freq_y[i, j], center_x-low_freq_x[i, j]:center_x + low_freq_x[i, j]] = True

        assert not center_mask.isnan().any()
        return masked_k_space * center_mask

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1) + 1e-6

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean
