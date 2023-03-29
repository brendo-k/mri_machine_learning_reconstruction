import torch.nn as nn
import torch
from ml_recon.Models import Unet, NormUnet, SensetivityModel
import einops
from ml_recon.Utils import fft_2d_img, ifft_2d_img, complex_conversion

class VarNet(nn.Module):
    def __init__(self, in_chan, out_chan, num_cascades=6, sens_chans=8, model_chans=18, use_norm=True, dropout_prob=0) -> None:
        super().__init__()
        self.cascade = nn.ModuleList(
            [VarnetBlock(Unet(in_chan, out_chan, chans=model_chans, with_instance_norm=use_norm, drop_prob=dropout_prob)) for _ in range(num_cascades)]
            )
        self.sens_model = SensetivityModel(in_chan, out_chan, chans=sens_chans, mask_center=True)
        self.lambda_reg = nn.Parameter(torch.ones((num_cascades)))

    # k-space sent in [B, C, H, W]
    def forward(self, k_space, mask):
        sense_maps = self.sens_model(k_space)
        cur_k = k_space.clone()
        for i, cascade in enumerate(self.cascade):
            refined_k = cascade(k_space, sense_maps)
            zero = torch.zeros(1, 1, 1, 1).to(cur_k)
            data_consistency = torch.where(mask, cur_k - refined_k, zero)
            cur_k = cur_k - self.lambda_reg[i] * data_consistency + refined_k
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
