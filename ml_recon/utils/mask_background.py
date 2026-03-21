import torch
import torch.nn.functional as F 
from torchvision.transforms.functional import gaussian_blur


def get_image_background_mask(ground_truth_image):
    # ground truth image shape b, con, h, w

    # gaussian blur image for better masking (blurring improves SNR)
    ground_truth_blurred = gaussian_blur(ground_truth_image, kernel_size=15, sigma=10.0) # type: ignore

    # get noise
    noise = ground_truth_blurred[..., :20, :20]
    # take the max value and scale up a bit
    mask_threshold = noise.amax((-1, -2)) * 1.20

    # same shape as image
    mask_threshold = mask_threshold.unsqueeze(-1).unsqueeze(-1)

    # get mask
    image_background_mask = ground_truth_blurred > mask_threshold 

    mask = dialate_mask(image_background_mask)

    # If there are any masks that are all zero, set to all 1s
    all_zero_masks_indecies = (~mask).all(dim=-1).all(dim=-1)
    # check if there are zero mask indexes
    if all_zero_masks_indecies.any():
        mask[all_zero_masks_indecies, :, :] = True

    return mask



def dialate_mask(mask, kernel_size=3):
    b, contrast, h, w = mask.shape
    mask = mask.view(b*contrast, h, w)
    dialed_mask = dilate(mask.to(torch.float32), kernel_size)
    return dialed_mask.to(torch.bool).view(b, contrast, h, w)


def dilate(image: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Applies morphological dilation to a 2D image tensor.

    Args:
        image (torch.Tensor): Input tensor of shape (B, H, W).
        kernel_size (int): Size of the square dilation kernel. Should be an odd number.

    Returns:
        torch.Tensor: Dilated tensor of shape (B, H, W).
    """
    if image.dim() != 3:
        raise ValueError("Input tensor must have shape (B, H, W)")

    # Convert (B, H, W) -> (B, 1, H, W) for compatibility with max_pool2d
    image = image.unsqueeze(1)

    # Apply max pooling to simulate dilation
    dilated = F.max_pool2d(image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    # Remove extra channel dimension
    return dilated.squeeze(1)