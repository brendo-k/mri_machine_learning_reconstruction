import torch.nn as nn
import torch
from typing import Tuple, Union
from functools import partial

from ml_recon.models import Unet
from functools import partial
from ml_recon.models import SensetivityModel_mc
from ml_recon.utils import fft_2d_img, ifft_2d_img, complex_conversion
from dataclasses import dataclass

@dataclass
class VarnetConfig:
    contrast_order: list
    cascades: int = 5
    sense_chans: int = 8
    channels: int = 18
    sensetivity_estimation: str = 'first' # can be first, joint, individual
    split_contrast_by_phase: bool = False


class MultiContrastVarNet(nn.Module):
    def __init__(self, config: VarnetConfig):
        super().__init__()
        self.config = config

        contrasts = len(self.config.contrast_order)
        model_backbone = partial(Unet, in_chan=contrasts*2, out_chan=contrasts*2, chans=self.config.channels)

        # module cascades
        self.cascades = nn.ModuleList(
            [VarnetBlock(model_backbone(), self.config.split_contrast_by_phase) for _ in range(self.config.cascades)]
        )

        # model to estimate sensetivities
        self.sens_model = SensetivityModel_mc(
            in_chans=2, 
            out_chans=2, 
            chans=self.config.sense_chans, 
            sensetivity_estimation=self.config.sensetivity_estimation,
            contrasts=contrasts
            )

        # regularizer weight
        self.lambda_reg = nn.Parameter(torch.ones(self.config.cascades, 1))

    # k-space sent in [B, C, H, W]
    def forward(self, undersampled_k, mask):
        # get sensetivity maps
        assert not torch.isnan(undersampled_k).any()
        assert not torch.isnan(mask).any()
        sense_maps = self.sens_model(undersampled_k, mask)

        assert not torch.isnan(sense_maps).any()

        # current k_space 
        current_k = undersampled_k.clone()
        for i, cascade in enumerate(self.cascades):
            # go through ith model cascade
            refined_k = cascade(current_k, sense_maps)
            assert not torch.isnan(undersampled_k).any()
            assert not torch.isnan(refined_k).any()

            data_consistency = mask * (current_k - undersampled_k)
            # gradient descent step
            current_regularization = self.lambda_reg[i]
            current_regularization = current_regularization[None, :, None, None, None]
            current_k = current_k - (current_regularization * data_consistency) - refined_k
        return current_k


class VarnetBlock(nn.Module):
    def __init__(self, model: nn.Module, split_complex_by_phase: bool = False) -> None:
        super().__init__()
        self.model = model
        if split_complex_by_phase:
            self.complex_forward = complex_conversion.complex_to_polar
            self.complex_backwards = complex_conversion.polar_to_complex
        else:
            self.complex_forward = complex_conversion.complex_to_real
            self.complex_backwards = complex_conversion.real_to_complex

    # sensetivities data [B, contrast, C, H, W]
    def forward(self, k_space, sensetivities):
        # Reduce (coil combine estimate to images)
        images = ifft_2d_img(k_space, axes=[-1, -2])
        images = torch.sum(images * sensetivities.conj(), dim=2)

        # Images after complex_forward [B, contrast * 2, h, w] (real)
        images = self.complex_forward(images)
        images, mean, std = self.norm(images)
        images = self.model(images)
        images = self.unnorm(images, mean, std)
        images = self.complex_backwards(images)

        # Expand
        images = sensetivities * images.unsqueeze(2)
        images = fft_2d_img(images, axes=[-1, -2])

        return images
    
    # is this not just instance norm?
    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # instance norm
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-9

        x = (x - mean) / std
        return x, mean, std


    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        x = x * std + mean
        return x
