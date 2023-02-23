import torch
import einops
# images [B, C, H, W] as complex. Conver to real -> [B, C, H, W, Complex(dim 2)]. Permute to [B, complex * C, H, W]
# Converts complex tensor to real tensor and concats the complex dimension to channels
def complex_to_real(images):
    # images dims [B, C, H, W, complex]
    images = torch.view_as_real(images)
    images = einops.rearrange(images, 'b c h w cm -> b (c cm) h w')
    return images

def real_to_complex(images):
    images = einops.rearrange(images, 'b (c cm) h w -> b c h w cm', cm=2)
    images = images.contiguous()
    images = torch.view_as_complex(images)
    return images