import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.utils.evaluation_functions import nmse, psnr


class plReconModel(pl.LightningModule):
    """This is a superclass for all reconstruction models. It tests the output 
    vs the ground truth using SSIM, PSNR, and NMSE. Handles multiple contrasts.
    Most recon networks here inhereit from this class. 
    """

    def __init__(self, contrast_order):
        super().__init__()
        self.contrast_order = contrast_order


    def test_step(self, batch, batch_index):
        estimate_k, k_space = batch

        estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)
        ground_truth_image = root_sum_of_squares(ifft_2d_img(k_space), coil_dim=2) 

        scaling_factor = ground_truth_image.amax((-1, -2), keepdim=True)
        image_background_mask = ground_truth_image > scaling_factor * 0.12

        estimated_image /= scaling_factor
        ground_truth_image /= scaling_factor

        estimated_image *= image_background_mask
        ground_truth_image *= image_background_mask

        average_ssim = 0
        average_psnr = 0
        average_nmse = 0
        for i in range(ground_truth_image.shape[0]):
            for contrast_index in range(len(self.contrast_order)):
                contrast_ground_truth = ground_truth_image[i, contrast_index, :, :]
                contrast_estimated = estimated_image[i, contrast_index, :, :]
                contrast_ground_truth = contrast_ground_truth[None, None, :, :]
                contrast_estimated = contrast_estimated[None, None, :, :]


                nmse_contrast = nmse(contrast_ground_truth, contrast_estimated)
                ssim_contrast, ssim_image = ssim(
                    contrast_ground_truth, 
                    contrast_estimated, 
                    return_full_image=True, 
                    data_range=(0, contrast_ground_truth.max().item())
                    )
                psnr_contrast = psnr(contrast_ground_truth, contrast_estimated, image_background_mask)

                # remove mask points that would equal to 1 (possibly some estimated points
                # will be removed here but only if matches completely in the kernel)
                                                        
                ssim_contrast = ssim_image[image_background_mask[i, contrast_index].unsqueeze(0).unsqueeze(0)].mean()

                self.log(f"metrics/nmse_" + self.contrast_order[contrast_index], nmse_contrast, sync_dist=True, on_step=True)
                self.log(f"metrics/ssim_torch_" + self.contrast_order[contrast_index], ssim_contrast, sync_dist=True, on_step=True)
                self.log(f"metrics/psnr_" + self.contrast_order[contrast_index], psnr_contrast, sync_dist=True, on_step=True)

                average_ssim += ssim_contrast
                average_psnr += psnr_contrast
                average_nmse += nmse_contrast


        average_nmse /= (ground_truth_image.shape[0] * ground_truth_image.shape[1])
        average_psnr /= (ground_truth_image.shape[0] * ground_truth_image.shape[1])
        average_ssim /= (ground_truth_image.shape[0] * ground_truth_image.shape[1])
        self.log(f'metrics/mean_ssim', average_ssim, on_epoch=True, sync_dist=True)
        self.log(f'metrics/mean_psnr', average_psnr, on_epoch=True, sync_dist=True)
        self.log(f'metrics/mean_nmse', average_nmse, on_epoch=True, sync_dist=True)
        
        return {
            'loss': 0,
            'estimate_image': estimated_image,
            'ground_truth_image': ground_truth_image,
            'mask': image_background_mask
        }

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        estimate_k, k_space = batch
        estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)
        ground_truth_image = root_sum_of_squares(ifft_2d_img(k_space), coil_dim=2) 

        scaling_factor = ground_truth_image.amax((-1, -2), keepdim=True)
        image_background_mask = ground_truth_image > scaling_factor * 0.1

        estimated_image /= scaling_factor
        ground_truth_image /= scaling_factor
        difference_image = (ground_truth_image - estimated_image).abs()

        estimated_image *= image_background_mask
        ground_truth_image *= image_background_mask

        
        estimated_image = estimated_image[0].clamp(0, 1)
        ground_truth_image = ground_truth_image[0]
        difference_image = (difference_image*10).clamp(0, 1)
        difference_image = difference_image[0]
        image_background_mask = image_background_mask[0]
        wandb_logger = self.logger
        if isinstance(wandb_logger, WandbLogger):
            wandb_logger.log_image(f'test/recon', self.convert_image_for_plotting(estimated_image))
            wandb_logger.log_image(f'test/target', self.convert_image_for_plotting(ground_truth_image))
            wandb_logger.log_image(f'test/diff', self.convert_image_for_plotting((difference_image)))
            wandb_logger.log_image(f'test/test_mask', self.convert_image_for_plotting(image_background_mask))


    def convert_image_for_plotting(self, image: torch.Tensor):
        contrasts = image.shape[0]
        return np.split(image.unsqueeze(1).cpu().numpy(), contrasts, 0)


    @torch.no_grad()
    def plot_images(self, estimate_k, k_space, mask, mode='train'):
        estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)
        fully_sampled_image = root_sum_of_squares(ifft_2d_img(k_space), coil_dim=2)
        
        # in ssdu target is other set
        estimated_image = estimated_image[0]/fully_sampled_image[0].amax((-1, -2), keepdim=True)
        fully_sampled_image = fully_sampled_image[0]/fully_sampled_image[0].amax((-1, -2), keepdim=True)
        diff = (estimated_image - fully_sampled_image).abs()*10
        k_space_scaled = k_space.abs()/(k_space.abs().max() / 50) 

        estimated_image = estimated_image.clamp(0, 1)
        fully_sampled_image = fully_sampled_image.clamp(0, 1)
        diff = diff.clamp(0, 1)
        k_space_scaled = k_space_scaled.clamp(0, 1)
        if self.logger and isinstance(self.logger, WandbLogger):
            wandb_logger = self.logger

            contrasts = estimated_image.shape[0]
            wandb_logger.log_image(mode + '/recon', np.split(estimated_image.unsqueeze(1).cpu().numpy(), contrasts, 0))
            wandb_logger.log_image(mode + '/fully_sampled', np.split(fully_sampled_image.unsqueeze(1).cpu().numpy(), contrasts, 0))
            wandb_logger.log_image(mode + '/mask', np.split(mask[0, :, [0], :, :].cpu().numpy(), contrasts, 0))
            wandb_logger.log_image(mode + '/diff', np.split(diff.unsqueeze(1).cpu().numpy(), contrasts, 0))
