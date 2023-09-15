import torch.nn as nn
import torch
from typing import Tuple

from ml_recon.models import SensetivityModel_mc
from ml_recon.utils import fft_2d_img, ifft_2d_img, complex_conversion


class VarNet_mc(nn.Module):
    def __init__(self, 
                 model_backbone: nn.Module,
                 num_cascades=6,
                 sens_chans=8,
                 ):
        super().__init__()

        # module cascades
        self.cascade = nn.ModuleList()

        for _ in range(num_cascades):
            self.cascade.append(
                VarnetBlock(
                    model_backbone()
                )
            )

        # model to estimate sensetivities
        self.sens_model = SensetivityModel_mc(2, 2, chans=sens_chans, mask_center=True)
        # regularizer weight
        self.lambda_reg = nn.Parameter(torch.ones((num_cascades)))

    # k-space sent in [B, C, H, W]
    def forward(self, reference_k, mask):
        # get sensetivity maps
        sense_maps = self.sens_model(reference_k, mask)
        # current k_space 
        current_k = reference_k.clone()
        for i, cascade in enumerate(self.cascade):
            # go through ith model cascade
            refined_k = cascade(current_k, sense_maps)
            data_consistency = self.data_consistency(current_k, reference_k, mask)
            # gradient descent step
            current_k = current_k - self.lambda_reg[i] * data_consistency - refined_k
        return current_k

    def data_consistency(self, current_k, reference_k, mask):
        # mask values
        zero = torch.zeros(1, 1, 1, 1).to(current_k).detach()
        # zero where not in mask
        dc_value = torch.where(mask, current_k - reference_k, zero)
        return dc_value

class VarnetBlock(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    # sensetivities data [B, C, H, W]
    def forward(self, images, sensetivities):
        # Reduce

        images = ifft_2d_img(images, axes=[-1, -2])
        combined_images = torch.sum(images * sensetivities.conj(), dim=2)

        batch, contrast, chan, height, width = images.shape
        print(images.shape)

        combined_images = combined_images.view(batch, contrast*chan, height, width)
        combined_images = complex_conversion.complex_to_real(combined_images)
        combined_images, mean, std = self.norm(combined_images)
        combined_images = self.model(combined_images)
        combined_images = self.unnorm(combined_images, mean, std)
        combined_images = complex_conversion.real_to_complex(combined_images)
        combined_images = combined_images.view(batch, contrast, chan, height, width)

        combined_images = complex_conversion.complex_to_real(combined_images)

        # Expand
        images = sensetivities * combined_images[:, :, None, :, :]
        images = fft_2d_img(images, axes=[-1, -2])

        return images
    
    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1).detach()
        std = x.std(dim=2).view(b, 2, 1, 1).detach()

        std = std + 1e-8

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean
