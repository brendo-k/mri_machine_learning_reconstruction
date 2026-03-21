import torch
import numpy as np
from typing import Union 

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger

import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.utils.evaluation_functions import nmse, psnr
from ml_recon.utils.mask_background import get_image_background_mask



class plReconModel(pl.LightningModule):
    """This is a superclass for all reconstruction models. It tests the output 
    vs the ground truth using SSIM, PSNR, and NMSE. Handles multiple contrasts.
    Most recon networks here inhereit from this class. 
    """

    def __init__(self, contrast_order, is_mask_testing=True, mask_threshold: Union[dict, None] = None):
        super().__init__()
        self.contrast_order = contrast_order
        self.is_mask_testing = is_mask_testing
        self.contrast_psnr_masked = [[] for _ in contrast_order]
        self.contrast_nmse_masked = [[] for _ in contrast_order]
        self.contrast_ssim_masked = [[] for _ in contrast_order]


    def my_test_step(self, batch, batch_index):
        estimate_k, ground_truth_image = batch
        if self.is_mask_testing:
            background_mask = get_image_background_mask(ground_truth_image)
        else:
            background_mask = torch.ones_like(ground_truth_image)

        estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)

        scaling_factor = ground_truth_image.amax((-1, -2), keepdim=True)

        estimated_image /= scaling_factor
        ground_truth_image = ground_truth_image / scaling_factor

        estimated_image_masked = estimated_image * background_mask
        ground_truth_image_masked = ground_truth_image * background_mask

        for contrast_index in range(len(self.contrast_order)):            
            for i in range(ground_truth_image.shape[0]):
                # get a slice of a contrast
                contrast_ground_truth_masked = ground_truth_image_masked[i, contrast_index, :, :]
                contrast_estimated_masked = estimated_image_masked[i, contrast_index, :, :]

                # reshape to proper shape for metrics
                contrast_ground_truth_masked = contrast_ground_truth_masked[None, None, :, :]
                contrast_estimated_masked = contrast_estimated_masked[None, None, :, :]
                
                # masked metrics
                nmse_val_masked = nmse(contrast_ground_truth_masked, contrast_estimated_masked)

                # ssim metrics
                ssim_val_masked = ssim(
                    contrast_ground_truth_masked, 
                    contrast_estimated_masked, 
                    data_range=(contrast_ground_truth_masked.max().item()),
                    kernel_size=7
                    )

                assert isinstance(ssim_val_masked, torch.Tensor)

                psnr_val_masked = psnr(contrast_ground_truth_masked, contrast_estimated_masked)


                self.contrast_ssim_masked[contrast_index].append(ssim_val_masked.cpu())
                self.contrast_psnr_masked[contrast_index].append(psnr_val_masked.cpu())
                self.contrast_nmse_masked[contrast_index].append(nmse_val_masked.cpu())

                self.log(f"metrics/masked_nmse_{self.contrast_order[contrast_index]}", nmse_val_masked, sync_dist=True, on_step=True)
                self.log(f"metrics/masked_ssim_{self.contrast_order[contrast_index]}", ssim_val_masked, sync_dist=True, on_step=True)
                self.log(f"metrics/masked_psnr_{self.contrast_order[contrast_index]}", psnr_val_masked, sync_dist=True, on_step=True)

        
        return {
            'loss': 0,
            'estimate_image': estimated_image,
            'ground_truth_image': ground_truth_image,
            'mask': background_mask
        }

    def on_test_end(self):
        for contrast_index, contrast_label in enumerate(self.contrast_order):
            nmse_masked_array = np.array(self.contrast_nmse_masked[contrast_index])
            ssim_masked_array = np.array(self.contrast_ssim_masked[contrast_index])
            psnr_masked_array = np.array(self.contrast_psnr_masked[contrast_index])

            print(f"metrics_mine/masked_nmse_{contrast_label}", nmse_masked_array.mean())
            print(f"metrics_mine/masked_ssim_{contrast_label}", ssim_masked_array.mean())
            print(f"metrics_mine/masked_psnr_{contrast_label}", psnr_masked_array.mean())


            print(f"metrics_mine/masked_nmse_std_{contrast_label}", nmse_masked_array.std())
            print(f"metrics_mine/masked_ssim_std_{contrast_label}", ssim_masked_array.std())
            print(f"metrics_mine/masked_psnr_std_{contrast_label}", psnr_masked_array.std())

            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log({f"metrics/nmse_masked_{contrast_label}_std": nmse_masked_array.std()})
                self.logger.experiment.log({f"metrics/ssim_masked_{contrast_label}_std": ssim_masked_array.std()})
                self.logger.experiment.log({f"metrics/psnr_masked_{contrast_label}_std": psnr_masked_array.std()})


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
        wandb_logger.log_image(f'test/{label}_diff', self.convert_image_for_plotting(difference_image))





    def convert_image_for_plotting(self, image: torch.Tensor):
        contrasts = image.shape[0]
        return np.split(image.unsqueeze(1).cpu().numpy(), contrasts, 0)
