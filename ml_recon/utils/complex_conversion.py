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

# Converts complex tensor to polar coordinates and concats the polar components to channels
def complex_to_polar(images: torch.Tensor):
    assert images.is_complex(), "Input tensor must be complex."
    # images dims [B, C, H, W]
    magnitude = torch.abs(images)  # Compute magnitude
    phase = torch.angle(images)   # Compute phase
    # Concatenate magnitude and phase along the channel dimension
    images = torch.cat((magnitude, phase), dim=1)  # [B, 2*C, H, W]
    return images

# Converts polar coordinates (magnitude and phase) back to a complex tensor
def polar_to_complex(images: torch.Tensor):
    assert images.size(1) % 2 == 0, "Channel dimension must be even (magnitude and phase pairs)."
    # images dims [B, 2*C, H, W]
    c = images.size(1) // 2
    magnitude, phase = images[:, :c, ...], images[:, c:, ...]  # Split into magnitude and phase
    # Reconstruct complex tensor
    complex_tensor = magnitude * torch.exp(1j * phase)
    return complex_tensor