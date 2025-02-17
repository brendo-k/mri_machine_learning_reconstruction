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
        images = images / rss_norm
        return images

    def mask_center(self, coil_k_spaces, mask):
        # I did some strange things here. Before, I tried to find the largest 2d box 
        # that was continuously contained in k-space. However, that added some extra 
        # bugs in self-suprvised trainign as the center box size could change depending
        # on the sets. I have just hard coded this for now, but could be interesting to 
        # test different coil estimation methods 
        acs_box_size = 10
        height = mask.shape[-2]
        width = mask.shape[-1]
        center_x = width // 2
        center_y = height // 2
        
        mask = torch.zeros((height, width), dtype=torch.bool, device=coil_k_spaces.device) 

        acs_slice_x = slice(center_x-acs_box_size//2, center_x+acs_box_size//2)
        mask[:, acs_slice_x] = 1

        return coil_k_spaces * mask
        

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