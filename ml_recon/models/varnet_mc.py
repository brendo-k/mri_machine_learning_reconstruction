import torch.nn as nn
import torch
from typing import Tuple, Union
from functools import partial

from ml_recon.models import Unet
from ml_recon.models import SensetivityModel_mc
from ml_recon.utils import fft_2d_img, ifft_2d_img, complex_conversion


class VarNet_mc(nn.Module):
    def __init__(self, 
                 model_backbone: Union[nn.Module, partial],
                 num_cascades=6,
                 sens_chans=8,
                 weight_sharing=False
                 ):
        super().__init__()

        # module cascades
        self.cascade = nn.ModuleList()
        
        if weight_sharing:
            assert isinstance(model, nn.Module), "Model is not an instance of nn.Module"
            self.cascades = nn.ModuleList(
                [VarnetBlock(model_backbone) for _ in range(num_cascades)]
            )
        else:
            self.cascades = nn.ModuleList(
                [VarnetBlock(model_backbone()) for _ in range(num_cascades)]
            )

        # model to estimate sensetivities
        self.sens_model = SensetivityModel_mc(2, 2, chans=sens_chans, mask_center=True)
        # regularizer weight
        self.lambda_reg = nn.Parameter(torch.ones((num_cascades)))

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
            current_k = current_k - (self.lambda_reg[i] * data_consistency) - refined_k
        return current_k


class VarnetBlock(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    # sensetivities data [B, contrast, C, H, W]
    def forward(self, images, sensetivities):
        # Reduce
        images = ifft_2d_img(images, axes=[-1, -2])

        # Images now [B, contrast, h, w] (complex)
        images = torch.sum(images * sensetivities.conj(), dim=2)

        # Images now [B, contrast * 2, h, w] (real)
        images = complex_conversion.complex_to_real(images)
        images, mean, std = self.norm(images)
        images = self.model(images)
        images = self.unnorm(images, mean, std)
        images = complex_conversion.real_to_complex(images)

        # Expand
        images = sensetivities * images[:, :, None, :, :]
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

