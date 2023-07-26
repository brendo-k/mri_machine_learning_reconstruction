import torch.nn as nn
import torch
from ml_recon.models import SensetivityModel
from ml_recon.models.resnet import ResNet
from ml_recon.utils import fft_2d_img, ifft_2d_img, complex_conversion


class VarNet(nn.Module):
    def __init__(self, 
                 in_chan=2, 
                 out_chan=2, 
                 num_cascades=6, 
                 sens_chans=8, 
                 model_chans=18, 
                 dropout_prob=0):
        super().__init__()
        # module cascades
        self.cascade = nn.ModuleList()
        for _ in range(num_cascades):
            self.cascade.append(
                VarnetBlock(
                    ResNet(9)
                )
            )

        # model to estimate sensetivities
        self.sens_model = SensetivityModel(in_chan, out_chan, chans=sens_chans, mask_center=True)
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
            # mask values
            zero = torch.zeros(1, 1, 1, 1).to(current_k)
            # zero where not in mask
            data_consistency = torch.where(mask.unsqueeze(1), current_k - reference_k, zero)
            # gradient descent step
            current_k = current_k - self.lambda_reg[i] * data_consistency - refined_k
        return current_k

class VarnetBlock(nn.Module):
    def __init__(self, unet: nn.Module) -> None:
        super().__init__()
        self.unet = unet

    # sensetivities data [B, C, H, W]
    def forward(self, images, sensetivities):
        # Reduce
        images = ifft_2d_img(images, axes=[2, 3])
        combined_images = torch.sum(images * sensetivities.conj_physical(), dim=1, keepdim=True)

        combined_images = complex_conversion.complex_to_real(combined_images)
        combined_images = self.unet(combined_images)
        combined_images = complex_conversion.real_to_complex(combined_images)

        # Expand
        images = sensetivities * combined_images
        images = fft_2d_img(images, axes=[2, 3])

        return images
    