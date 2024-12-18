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
            batch_contrasts: bool = False,
            sensetivity_estimation: str = 'first'
            ):

        """Module used to estimate sensetivity maps based on masked center k-space

        Args:
            in_chans (int): _description_
            out_chans (int): _description_
            chans (int): Number of convolutional channels for U-Net estimator
            mask_center (bool, optional): Mask k-space for estimation. Defaults to True.
            batch_contrasts (bool, optional): Batch contrasts or pass along channel dimension. Defaults to False.
        """
        super().__init__()
        self.batch_contrasts = batch_contrasts
        self.sensetivity_estimation = sensetivity_estimation
        self.model = Unet(in_chans, out_chans, chans=chans)

    # recieve coil maps as [B, contrast, channels, H, W]
    def forward(self, images, mask):
        images = self.mask_center(images, mask) 
        # get the first image for estimating coil sensetivites
        if self.sensetivity_estimation == 'first':
            images = images[:, [0], :, :, :]

        images = ifft_2d_img(images, axes=[-1, -2])
        assert isinstance(images, torch.Tensor)

        number_of_coils = images.shape[2]
        num_contrasts = images.shape[1]

        if self.sensetivity_estimation == 'joint':
            images = images.swapdims(1, 2) # b contrast, chan, h, w -> b chan con h w
            images = einops.rearrange(images, 'b c contrast h w -> (b c) contrast h w')
        elif self.sensetivity_estimation == 'first' or self.sensetivity_estimation == 'independent':
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
        if self.sensetivity_estimation == 'joint':
            images = einops.rearrange(images, '(b c) contrast h w -> b c contrast h w', c=number_of_coils, contrast=num_contrasts)
            images = images.swapdims(1, 2) # swap back
        elif self.sensetivity_estimation == 'first' or self.sensetivity_estimation == 'independent':
            images = einops.rearrange(images, '(b contrast c) 1 h w -> b contrast c h w', c=number_of_coils, contrast=num_contrasts)
        # rss to normalize sense maps
        rss_norm = root_sum_of_squares(images, coil_dim=2).unsqueeze(2) 
        #assert not (rss_norm == 0).any()
        images = images / rss_norm
        return images

    def mask_center(self, coil_k_spaces, mask):
        # coil_k: [b cont chan height width]
        masked_k_space = coil_k_spaces.clone()

        center_x = mask.shape[-1] // 2
        center_y = mask.shape[-2] // 2
        
        # Get the squezed masks in vertical and horizontal directions (batch, contrast, PE or FE)
        squeezed_mask_hor = (mask[:, :, 0, center_y, :]).to(torch.int8)
        squeezed_mask_vert = (mask[:, :, 0, :, center_x]).to(torch.int8)

        # Get the first zero index starting from the center. (TODO: This is a problem if they are all zeros or ones...)
        left = torch.argmin(squeezed_mask_hor[..., :center_x].flip(-1), dim=-1)
        right = torch.argmin(squeezed_mask_hor[..., center_x:], dim=-1)
        top = torch.argmin(squeezed_mask_vert[..., :center_y].flip(-1), dim=-1)
        bottom = torch.argmin(squeezed_mask_vert[..., center_y:], dim=-1)


        # if phase encoding lines, acquire whole line for acs calculations
        if (top == 0).all():
            top = torch.full(top.shape, center_y)
            bottom = torch.full(top.shape, center_y)

        # force symmetric left and right acs boundries
        low_freq_x = torch.min(left, right)
        low_freq_y = torch.min(top, bottom)
        low_freq_x[low_freq_x < 5] = 5
        low_freq_y[low_freq_y < 5] = 5
        center_mask = torch.zeros_like(masked_k_space[:, :, 0, :, :], dtype=torch.bool)
        # loop through num_low freq tensor and set acs lines to true
        for i in range(low_freq_x.shape[0]):
            for j in range(low_freq_y.shape[1]):
                center_mask[i, j, 
                            center_y - low_freq_y[i, j]:center_y + low_freq_y[i, j], 
                            center_x - low_freq_x[i, j]:center_x + low_freq_x[i, j]
                            ] = True

        return masked_k_space * center_mask.unsqueeze(2)

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean