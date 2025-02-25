import torch
import numpy as np
from typing import Union 

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger

import torch.nn.functional as F 
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchvision.transforms.functional import gaussian_blur


from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.utils.evaluation_functions import nmse, psnr



class plReconModel(pl.LightningModule):
    """This is a superclass for all reconstruction models. It tests the output 
    vs the ground truth using SSIM, PSNR, and NMSE. Handles multiple contrasts.
    Most recon networks here inhereit from this class. 
    """

    def __init__(self, contrast_order, is_mask_testing=True, mask_threshold: Union[dict, None] = None):
        super().__init__()
        self.contrast_order = contrast_order
        self.is_mask_testing = is_mask_testing
        self.mask_threshold = mask_threshold


    def test_step(self, batch, batch_index):
        estimate_k, ground_truth_image = batch

        estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)

        scaling_factor = ground_truth_image.amax((-1, -2), keepdim=True)

        estimated_image /= scaling_factor
        ground_truth_image = ground_truth_image / scaling_factor

        background_mask = self.get_image_background_mask(ground_truth_image)
        estimated_image = estimated_image * background_mask
        ground_truth_image = ground_truth_image * background_mask

        average_ssim = 0
        average_psnr = 0
        average_nmse = 0
        for contrast_index in range(len(self.contrast_order)):            
            contrast_psnr = 0
            contrast_nmse = 0
            contrast_ssim = 0
            for i in range(ground_truth_image.shape[0]):
                contrast_ground_truth = ground_truth_image[i, contrast_index, :, :]
                contrast_estimated = estimated_image[i, contrast_index, :, :]
                contrast_mask = background_mask[i, contrast_index, :, :]
                contrast_mask = contrast_mask[None, None, :, :]
                contrast_ground_truth = contrast_ground_truth[None, None, :, :]
                contrast_estimated = contrast_estimated[None, None, :, :]


                nmse_val = nmse(contrast_ground_truth, contrast_estimated)
                ssim_val, ssim_image = ssim(
                    contrast_ground_truth, 
                    contrast_estimated, 
                    data_range=(0, contrast_ground_truth.max().item()),
                    return_full_image=True
                    )
                
                ssim_val = ssim_image[contrast_mask]
                ssim_val = ssim_val.mean()
                assert isinstance(ssim_val, torch.Tensor)          
                psnr_val = psnr(contrast_ground_truth, contrast_estimated)
                
                contrast_ssim += ssim_val
                contrast_psnr += psnr_val
                contrast_nmse += nmse_val

            contrast_ssim /= ground_truth_image.shape[0]
            contrast_nmse /= ground_truth_image.shape[0]
            contrast_psnr /= ground_truth_image.shape[0]
            self.log(f"metrics/nmse_{self.contrast_order[contrast_index]}", contrast_nmse, sync_dist=True, on_step=True)
            self.log(f"metrics/ssim_{self.contrast_order[contrast_index]}", contrast_ssim, sync_dist=True, on_step=True)
            self.log(f"metrics/psnr_{self.contrast_order[contrast_index]}", contrast_psnr, sync_dist=True, on_step=True)

            average_ssim += contrast_ssim
            average_psnr += contrast_psnr
            average_nmse += contrast_nmse

        average_nmse /= ground_truth_image.shape[1]
        average_psnr /= ground_truth_image.shape[1]
        average_ssim /= ground_truth_image.shape[1]
        self.log(f'metrics/mean_ssim', average_ssim, on_epoch=True, sync_dist=True)
        self.log(f'metrics/mean_psnr', average_psnr, on_epoch=True, sync_dist=True)
        self.log(f'metrics/mean_nmse', average_nmse, on_epoch=True, sync_dist=True)
        
        return {
            'loss': 0,
            'estimate_image': estimated_image,
            'ground_truth_image': ground_truth_image,
            'mask': background_mask
        }


    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        estimate_k, ground_truth_image = batch
        estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)

        scaling_factor = ground_truth_image.amax((-1, -2), keepdim=True)
        image_background_mask = self.get_image_background_mask(ground_truth_image)

        estimated_image /= scaling_factor
        ground_truth_image /= scaling_factor

        estimated_image *= image_background_mask
        ground_truth_image *= image_background_mask

        difference_image = (ground_truth_image - estimated_image).abs()
        
        estimated_image = estimated_image[0].clamp(0, 1)
        ground_truth_image = ground_truth_image[0]
        difference_image = (difference_image*10).clamp(0, 1)
        difference_image = difference_image[0]
        image_background_mask = image_background_mask[0]
        if isinstance(self.logger, WandbLogger):
            self.plot_test_images(ground_truth_image, estimated_image, image_background_mask, difference_image)

    def plot_test_images(self, ground_truth_image, estimated_image, image_background_mask, difference_image):
        wandb_logger = self.logger
        assert isinstance(wandb_logger, WandbLogger)
        
        wandb_logger.log_image(f'test/recon', self.convert_image_for_plotting(estimated_image))
        wandb_logger.log_image(f'test/target', self.convert_image_for_plotting(ground_truth_image))
        wandb_logger.log_image(f'test/diff', self.convert_image_for_plotting((difference_image)))
        wandb_logger.log_image(f'test/test_mask', self.convert_image_for_plotting(image_background_mask))


    def get_image_background_mask(self, ground_truth_image):
        if not self.is_mask_testing:
            return torch.ones_like(ground_truth_image)
        img_max = ground_truth_image.amax((-1, -2), keepdim=True)
        mask_threshold = []
        for contrast in self.contrast_order:
            if self.mask_threshold and contrast.lower() in self.mask_threshold:
                mask_threshold.append(self.mask_threshold[contrast.lower()])
            else:
                mask_threshold.append(0.1)
        
        mask_threshold = torch.tensor(mask_threshold, device=ground_truth_image.device)
        mask_threshold = mask_threshold.unsqueeze(-1).unsqueeze(-1)
        
        ground_truth_blurred = gaussian_blur(ground_truth_image, kernel_size=15, sigma=25.0) # type: ignore
        
        image_background_mask = ground_truth_blurred > img_max * mask_threshold 
        mask =  self.dialate_mask(image_background_mask)
        
        return mask


    
    def dialate_mask(self, mask, kernel_size=7):
        b, contrast, h, w = mask.shape
        mask = mask.view(b*contrast, h, w)
        dialed_mask = self.dilate(mask.to(torch.float32), kernel_size)
        return dialed_mask.to(torch.bool).view(b, contrast, h, w)
    
    
    def dilate(self, image: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
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

    
    def convert_image_for_plotting(self, image: torch.Tensor):
        contrasts = image.shape[0]
        return np.split(image.unsqueeze(1).cpu().numpy(), contrasts, 0)