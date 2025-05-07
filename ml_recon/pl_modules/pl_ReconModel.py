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


    def my_test_step(self, batch, batch_index, label=''):
        estimate_k, ground_truth_image = batch
        background_mask = self.get_image_background_mask(ground_truth_image)

        estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)

        scaling_factor = ground_truth_image.amax((-1, -2), keepdim=True)

        estimated_image /= scaling_factor
        ground_truth_image = ground_truth_image / scaling_factor

        estimated_image_masked = estimated_image * background_mask
        ground_truth_image_masked = ground_truth_image * background_mask

        average_ssim = 0
        average_ssim_masked = 0
        average_psnr_masked = 0
        average_nmse_masked = 0
        average_ssim = 0
        average_psnr = 0
        average_nmse = 0
        for contrast_index in range(len(self.contrast_order)):            
            contrast_psnr_masked = []
            contrast_nmse_masked = []
            contrast_ssim_masked = []
            contrast_psnr = []
            contrast_nmse = []
            contrast_ssim = []
            for i in range(ground_truth_image.shape[0]):
                # get a slice of a contrast
                contrast_ground_truth = ground_truth_image[i, contrast_index, :, :]
                contrast_ground_truth_masked = ground_truth_image_masked[i, contrast_index, :, :]
                contrast_estimated = estimated_image[i, contrast_index, :, :]
                contrast_estimated_masked = estimated_image_masked[i, contrast_index, :, :]

                # reshape to proper shape for metrics
                contrast_ground_truth = contrast_ground_truth[None, None, :, :]
                contrast_estimated = contrast_estimated[None, None, :, :]
                contrast_ground_truth_masked = contrast_ground_truth_masked[None, None, :, :]
                contrast_estimated_masked = contrast_estimated_masked[None, None, :, :]
                
                # masked metrics
                nmse_val = nmse(contrast_ground_truth, contrast_estimated)
                nmse_val_masked = nmse(contrast_ground_truth_masked, contrast_estimated_masked)

                # ssim metrics
                ssim_val_masked = ssim(
                    contrast_ground_truth_masked, 
                    contrast_estimated_masked, 
                    data_range=(0, contrast_ground_truth_masked.max().item()),
                    kernel_size=7
                    )
                ssim_val  = ssim(
                    contrast_ground_truth, 
                    contrast_estimated, 
                    data_range=(0, contrast_ground_truth.max().item()),
                    kernel_size=7
                    )

                assert isinstance(ssim_val, torch.Tensor)
                assert isinstance(ssim_val_masked, torch.Tensor)

                psnr_val_masked = psnr(contrast_ground_truth_masked, contrast_estimated_masked)
                psnr_val = psnr(contrast_ground_truth, contrast_estimated)
                
                contrast_ssim.append(ssim_val)
                contrast_psnr.append(psnr_val)
                contrast_nmse.append(nmse_val)

                contrast_ssim_masked.append(ssim_val_masked)
                contrast_psnr_masked.append(psnr_val_masked)
                contrast_nmse_masked.append(nmse_val_masked)

            contrast_ssim = np.array(contrast_ssim)
            contrast_psnr = np.array(contrast_psnr)            
            contrast_nmse = np.array(contrast_nmse)

            contrast_ssim_masked = np.array(contrast_ssim_masked)
            contrast_psnr_masked = np.array(contrast_psnr_masked)
            contrast_nmse_masked = np.array(contrast_nmse_masked)


            self.log(f"metrics/nmse_{self.contrast_order[contrast_index]}", contrast_nmse.mean(), sync_dist=True, on_step=True)
            self.log(f"metrics/ssim_{self.contrast_order[contrast_index]}", contrast_ssim.mean(), sync_dist=True, on_step=True)
            self.log(f"metrics/psnr_{self.contrast_order[contrast_index]}", contrast_psnr.mean(), sync_dist=True, on_step=True)

            self.log(f"metrics/nmse_std_{self.contrast_order[contrast_index]}", contrast_nmse.std(), sync_dist=True, on_step=True)
            self.log(f"metrics/ssim_std_{self.contrast_order[contrast_index]}", contrast_ssim.std(), sync_dist=True, on_step=True)
            self.log(f"metrics/psnr_std_{self.contrast_order[contrast_index]}", contrast_psnr.std(), sync_dist=True, on_step=True)

            self.log(f"metrics/masked_nmse_std_{self.contrast_order[contrast_index]}", contrast_nmse_masked.mean(), sync_dist=True, on_step=True)
            self.log(f"metrics/masked_ssim_std_{self.contrast_order[contrast_index]}", contrast_ssim_masked.mean(), sync_dist=True, on_step=True)
            self.log(f"metrics/masked_psnr_std_{self.contrast_order[contrast_index]}", contrast_psnr_masked.mean(), sync_dist=True, on_step=True)

            self.log(f"metrics/masked_nmse_std_{self.contrast_order[contrast_index]}", contrast_nmse_masked.std(), sync_dist=True, on_step=True)
            self.log(f"metrics/masked_ssim_std_{self.contrast_order[contrast_index]}", contrast_ssim_masked.std(), sync_dist=True, on_step=True)
            self.log(f"metrics/masked_psnr_std_{self.contrast_order[contrast_index]}", contrast_psnr_masked.std(), sync_dist=True, on_step=True)

            average_ssim += contrast_ssim.mean()
            average_psnr += contrast_psnr.mean()
            average_nmse += contrast_nmse.mean()

            average_ssim_masked += contrast_ssim_masked.mean()
            average_psnr_masked += contrast_psnr_masked.mean()
            average_nmse_masked += contrast_nmse_masked.mean()

        average_nmse /= ground_truth_image.shape[1]
        average_psnr /= ground_truth_image.shape[1]
        average_ssim /= ground_truth_image.shape[1]

        average_nmse_masked /= ground_truth_image.shape[1]
        average_psnr_masked /= ground_truth_image.shape[1]
        average_ssim_masked /= ground_truth_image.shape[1]

        self.log('metrics/mean_ssim', average_ssim, on_epoch=True, sync_dist=True)
        self.log('metrics/mean_psnr', average_psnr, on_epoch=True, sync_dist=True)
        self.log('metrics/mean_nmse', average_nmse, on_epoch=True, sync_dist=True)

        self.log('metrics/masked_mean_ssim', average_ssim_masked, on_epoch=True, sync_dist=True)
        self.log('metrics/masked_mean_psnr', average_psnr_masked, on_epoch=True, sync_dist=True)
        self.log('metrics/masked_mean_nmse', average_nmse_masked, on_epoch=True, sync_dist=True)
        
        return {
            'loss': 0,
            'estimate_image': estimated_image,
            'ground_truth_image': ground_truth_image,
            'mask': background_mask
        }


    def my_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        estimate_k, ground_truth_image = batch
        estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)

        scaling_factor = ground_truth_image.amax((-1, -2), keepdim=True)
        image_background_mask = self.get_image_background_mask(ground_truth_image)

        estimated_image /= scaling_factor
        ground_truth_image /= scaling_factor

        estimated_image_masked = estimated_image * image_background_mask
        ground_truth_image_masked = estimated_image * image_background_mask

        difference_image = (ground_truth_image - estimated_image).abs()
        difference_image_masked = (ground_truth_image_masked - estimated_image_masked).abs()
        
        estimated_image = estimated_image[0].clamp(0, 1)
        ground_truth_image = ground_truth_image[0]
        difference_image = (difference_image*10).clamp(0, 1)
        difference_image = difference_image[0]
        image_background_mask = image_background_mask[0]
        if isinstance(self.logger, WandbLogger):
            self.plot_test_images(
                ground_truth_image_masked, 
                estimated_image_masked, 
                difference_image_masked,
                label='masked',
                )
            self.plot_test_images(
                ground_truth_image, 
                estimated_image, 
                difference_image,
                label='unmasked',
                )

    def plot_test_images(
        self, 
        ground_truth_image, 
        estimated_image, 
        difference_image,
        label='',
        ):
        wandb_logger = self.logger
        assert isinstance(wandb_logger, WandbLogger)
        
        wandb_logger.log_image(f'test/{label}_recon', self.convert_image_for_plotting(estimated_image))
        wandb_logger.log_image(f'test/{label}_target', self.convert_image_for_plotting(ground_truth_image))
        wandb_logger.log_image(f'test/{label}_diff', self.convert_image_for_plotting((difference_image)))


    def get_image_background_mask(self, ground_truth_image):
        # ground truth image shape b, con, h, w
        if not self.is_mask_testing:
            return torch.ones_like(ground_truth_image)
        
        # get noise
        noise = ground_truth_image[..., :20, :20]
        # take the max value and scale up a bit
        mask_threshold = noise.amax((-1, -2)) * 1.05

        # same shape as image
        mask_threshold = mask_threshold.unsqueeze(-1).unsqueeze(-1)
        
        # gaussian blur image for better masking (blurring improves SNR)
        ground_truth_blurred = gaussian_blur(ground_truth_image, kernel_size=31, sigma=25.0) # type: ignore
        
        # get mask
        image_background_mask = ground_truth_blurred > mask_threshold 
        
        mask =  self.dialate_mask(image_background_mask)
        
        # If there are any masks that are all zero, set to all 1s
        all_zero_masks_indecies = (~mask ).all(dim=(-1, -2))
        # check if there are zero mask indexes
        if all_zero_masks_indecies.any():
            mask[all_zero_masks_indecies, :, :] = True
        
        return mask


    
    def dialate_mask(self, mask, kernel_size=5):
        
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
