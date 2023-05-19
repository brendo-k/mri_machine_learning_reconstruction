import torch.nn as nn
import torch
from ml_recon.models import Unet, NormUnet, SensetivityModel
from ml_recon.utils import fft_2d_img, ifft_2d_img, complex_conversion


class VarNet(nn.Module):
    def __init__(self, 
                 in_chan=2, 
                 out_chan=2, 
                 num_cascades=6, 
                 sens_chans=8, 
                 model_chans=18, 
                 use_norm=True, 
                 dropout_prob=0):
        super().__init__()
        # module cascades
        self.cascade = nn.ModuleList()
        for _ in range(num_cascades):
            self.cascade.append(
                VarnetBlock(
                    NormUnet(
                        in_chan,
                        out_chan,
                        chans=model_chans,
                        with_instance_norm=use_norm,
                        drop_prob=dropout_prob
                    )
                )
            )

        # model to estimate sensetivities
        self.sens_model = SensetivityModel(in_chan, out_chan, chans=sens_chans, mask_center=True)
        # regularizer weight
        self.lambda_reg = nn.Parameter(torch.ones((num_cascades)))

    # k-space sent in [B, C, H, W]
    def forward(self, k_ref, mask):
        # get sensetivity maps
        sense_maps = self.sens_model(k_ref)
        # current k_space 
        cur_k = k_ref.clone()
        for i, cascade in enumerate(self.cascade):
            # go through ith model cascade
            refined_k = cascade(cur_k, sense_maps)
            # mask values
            zero = torch.zeros(1, 1, 1, 1).to(cur_k)
            # zero where not in mask
            data_consistency = torch.where(mask.unsqueeze(1), cur_k - k_ref, zero)
            # gradient descent step
            cur_k = cur_k - self.lambda_reg[i] * data_consistency - refined_k
        return cur_k

class VarnetBlock(nn.Module):
    def __init__(self, unet: Unet) -> None:
        super().__init__()
        self.unet = unet

    # sensetivities data [B, C, H, W]
    def forward(self, images, sensetivities):
        images = ifft_2d_img(images, axes=[2, 3])
        combined_images = torch.sum(images * sensetivities.conj(), dim=1, keepdim=True)
        combined_images = complex_conversion.complex_to_real(combined_images)
        combined_images = self.unet(combined_images)
        combined_images = complex_conversion.real_to_complex(combined_images)
        images = sensetivities * combined_images
        images = fft_2d_img(images, axes=[2, 3])
        return images
