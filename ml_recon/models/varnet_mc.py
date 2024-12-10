import torch.nn as nn
import torch
from typing import Tuple, Union
from functools import partial

from ml_recon.models import Unet
from functools import partial
from ml_recon.models import SensetivityModel_mc
from ml_recon.utils import fft_2d_img, ifft_2d_img, complex_conversion


class VarNet_mc(nn.Module):
    def __init__(self, 
                 model_backbone: Union[nn.Module, partial, None] = None,
                 contrasts:int = 1,
                 num_cascades:int = 6,
                 sens_chans:int = 8,
                 chans:int = 32,
                 split_complex_by_phase: bool = True
                 ):
        super().__init__()

        if not model_backbone:
            model_backbone = partial(Unet, in_chan=contrasts*2, out_chan=contrasts*2, chans=chans)

        assert model_backbone is not None

        # module cascades
        self.cascades = nn.ModuleList(
            [VarnetBlock(model_backbone(), split_complex_by_phase) for _ in range(num_cascades)]
        )

        # model to estimate sensetivities
        self.sens_model = SensetivityModel_mc(2, 2, chans=sens_chans, mask_center=True)
        # regularizer weight

        self.lambda_reg = nn.Parameter(torch.ones(num_cascades, 1))

    # k-space sent in [B, C, H, W]
    def forward(self, reference_k, mask):
        # get sensetivity maps
        assert not torch.isnan(reference_k).any()
        assert not torch.isnan(mask).any()
        sense_maps = self.sens_model(reference_k, mask)

        assert not torch.isnan(sense_maps).any()

        # current k_space 
        current_k = reference_k.clone()
        for i, cascade in enumerate(self.cascades):
            # go through ith model cascade
            refined_k = cascade(current_k, sense_maps)
            assert not torch.isnan(reference_k).any()
            assert not torch.isnan(refined_k).any()

            data_consistency = mask * (current_k - reference_k)
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