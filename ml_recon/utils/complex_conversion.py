import torch
import einops
# images [B, C, H, W] as complex. Conver to real -> [B, C, H, W, Complex(dim 2)]. Permute to [B, complex * C, H, W]
# Converts complex tensor to real tensor and concats the complex dimension to channels
def complex_to_real(images: torch.Tensor):
    assert images.is_complex(), 'Channel dimension should be at least 2'
    # images dims [B, C, H, W, complex]
    images = torch.view_as_real(images)
    images = einops.rearrange(images, 'b c h w cm -> b (cm c) h w')
    return images

def real_to_complex(images: torch.Tensor):
    assert images.shape[1] >= 2, 'Channel dimension should be at least 2'
    images = einops.rearrange(images, 'b (cm c) h w -> b c h w cm', cm=2)
    images = images.contiguous()
    images = torch.view_as_complex(images)
    return images
