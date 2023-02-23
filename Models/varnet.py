import torch.nn as nn
import torch
from .SensetivityModel import SensetivityModel
from Models import Unet
import einops
from Utils import fft_2d_img, ifft_2d_img, complex_conversion

class VarNet(nn.Module):
    def __init__(self, in_chan, out_chan, num_cascades=6, sens_chans=8, model_chans=18, use_norm=False, dropout_prob=0) -> None:
        super().__init__()
        self.cascade = nn.ModuleList(
            [VarnetBlock(Unet(in_chan, out_chan, chans=model_chans, with_instance_norm=use_norm, drop_prob=dropout_prob)) for _ in range(num_cascades)]
            )
        self.sens_model = SensetivityModel(in_chan, out_chan, chans=sens_chans, mask_center=True)
        self.lambda_reg = nn.Parameter(torch.rand(1))

    # k-space sent in [B, C, H, W]
    def forward(self, k_space, mask):
        sense_maps = self.sens_model(k_space)
        cur_k = k_space
        for cascade in self.cascade:
            refined_k = cascade(k_space, sense_maps)
            data_consistency = mask * (cur_k - k_space)
            cur_k = cur_k - self.lambda_reg * data_consistency + refined_k

class VarnetBlock(nn.Module):
    def __init__(self, unet: Unet) -> None:
        super().__init__()
        self.unet = unet

    # sensetivities data [B, C, H, W]
    def forward(self, k_space_data, sensetivities):
        images = ifft_2d_img(k_space_data, axes=[2, 3])
        combined_images = torch.sum(images * sensetivities, dim=1, keepdim=True)
        combined_image_real = complex_conversion.complex_to_real(combined_images)
        refined_image = self.unet(combined_image_real)
        refined_image = complex_conversion.real_to_complex(refined_image)
        expanded = sensetivities * refined_image
        expanded = fft_2d_img(expanded)
        return expanded

    # images [B, C, H, W] as complex. Conver to real -> [B, C, H, W, Complex(dim 2)]. Permute to [B, complex * C, H, W]
    # Converts complex tensor to real tensor and concats the complex dimension to channels
    def complex_to_real(self, images):
        # images dims [B, C, H, W, complex]
        images = torch.view_as_real(images)
        images = einops.rearrange(images, 'b c h w cm -> b (c cm) h w')
        return images

    def real_to_complex(self, images):
        images = einops.rearrange(images, 'b (c cm) h w -> b c h w cm', cm=2)
        return images
