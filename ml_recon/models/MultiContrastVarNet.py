from typing import Tuple, Literal
from functools import partial
from dataclasses import dataclass

import torch.nn as nn
import torch

from ml_recon.models import Unet
from ml_recon.models import SensetivityModel_mc
from ml_recon.utils import fft_2d_img, ifft_2d_img, complex_to_real, real_to_complex

@dataclass
class VarnetConfig:
    contrast_order: list
    model: str = 'unet'
    cascades: int = 5
    sense_chans: int = 8
    channels: int = 18
    dropout: float = 0
    depth: int = 4
    upsample_method: Literal['conv', 'bilinear', 'max'] = 'conv'
    conv_after_upsample: bool = False


class MultiContrastVarNet(nn.Module):
    def __init__(self, config: VarnetConfig):
        super().__init__()
        contrasts = len(config.contrast_order)
        model_backbone = partial(
            Unet, 
            in_chan=contrasts*2, 
            out_chan=contrasts*2, 
            chans=config.channels, 
            drop_prob=config.dropout, 
            depth=config.depth, 
            upsample_method=config.upsample_method,
            conv_after_upsample=config.conv_after_upsample
        )

        # module cascades
        self.cascades = nn.ModuleList(
            [VarnetBlock(model_backbone()) for _ in range(config.cascades)]
        )

        # model to estimate sensetivities
        self.sens_model = SensetivityModel_mc(
            2, 
            2, 
            chans=config.sense_chans, 
            upsample_method=config.upsample_method, 
            conv_after_upsample=config.conv_after_upsample
            )
        # regularizer weight
        self.lambda_reg = nn.Parameter(torch.ones(config.cascades))

    # k-space sent in [B, C, H, W]
    def forward(self, reference_k, mask, zf_mask):
        # get sensetivity maps
        assert not torch.isnan(reference_k).any(), reference_k
        assert not torch.isnan(mask).any(), mask
        sense_maps = self.sens_model(reference_k, mask)

        assert not torch.isnan(sense_maps).any(), sense_maps
        
        # current k_space 
        current_k = reference_k.clone()
        for i, cascade in enumerate(self.cascades):
            current_k = current_k * zf_mask
            # go through ith model cascade
            refined_k = cascade(current_k, sense_maps)
            assert not torch.isnan(reference_k).any(), reference_k
            assert not torch.isnan(refined_k).any(), refined_k

            data_consistency = mask * (current_k - reference_k)
            # gradient descent step
            current_k = current_k - (self.lambda_reg[i] * data_consistency) - refined_k
        return current_k


class VarnetBlock(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


    # sensetivities data [B, contrast, C, H, W]
    def forward(self, k_space, sensetivities):
        assert not torch.isnan(k_space).any(), k_space
        # Reduce
        images = ifft_2d_img(k_space, axes=[-1, -2])

        # Images now [B, contrast, h, w] (complex)
        images = torch.sum(images * sensetivities.conj(), dim=2)

        # Images now [B, contrast * 2, h, w] (real)
        images = complex_to_real(images)
        images, mean, std = self.norm(images)
        assert not torch.isnan(images).any(), images
        images = self.model(images)
        assert not torch.isnan(images).any(), images
        images = self.unnorm(images, mean, std)
        assert not torch.isnan(images).any(), images
        images = real_to_complex(images)

        # Expand
        images = sensetivities * images.unsqueeze(2)
        images = fft_2d_img(images, axes=[-1, -2])

        return images

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # instance norm
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)

        x = (x - mean) / std
        return x, mean, std


    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        x = x * std + mean
        return x

