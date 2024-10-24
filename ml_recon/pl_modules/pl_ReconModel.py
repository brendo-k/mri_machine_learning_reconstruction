import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.utils.evaluate import nmse, psnr


class plReconModel(pl.LightningModule):
    """This is a superclass for all reconstruction models. It tests the output 
    vs the ground truth using SSIM, PSNR, and NMSE. Handles multiple contrasts.
    Most recon networks here inhereit from this class. 
    """

    def __init__(self, contrast_order):
        super().__init__()
        self.contrast_order = contrast_order


    def test_step(self, batch, batch_index):
        estimate_k, k_space, label = batch

        estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)
        ground_truth_image = root_sum_of_squares(ifft_2d_img(k_space), coil_dim=2) 
        scaling_factor = ground_truth_image.amax((-1, -2), keepdim=True)
        mask = ground_truth_image > 0.025

        estimated_image /= scaling_factor
        ground_truth_image /= scaling_factor
        estimated_image *= mask
        ground_truth_image *= mask
        diff = (ground_truth_image - estimated_image).abs()

        wandb_logger = self.logger
        contrasts = estimated_image.shape[1]

        if batch_index % 20 == 0:
            current_step = batch_index * estimated_image.shape[0]
            for i in range(estimated_image.shape[0]):
                wandb_logger.log_image(f'test/{label}_recon', np.split(np.clip(estimated_image[i].unsqueeze(1).cpu().numpy(), 0, 1), contrasts, 0), step=current_step+i)
                wandb_logger.log_image(f'test/{label}_target', np.split(ground_truth_image[i].unsqueeze(1).cpu().numpy(), contrasts, 0), step=current_step + i)
                wandb_logger.log_image(f'test/{label}_diff', np.split(np.clip(diff[i].unsqueeze(1).cpu().numpy()*4, 0, 1), contrasts, 0), step=current_step + i)
                wandb_logger.log_image(f'test/{label}_test_mask', np.split(mask[i].unsqueeze(1).cpu().numpy(), contrasts, 0), step=current_step+i)

        total_ssim = 0
        total_psnr = 0
        total_nmse = 0
        for contrast_index in range(len(self.contrast_order)):
            batch_nmse = 0
            batch_ssim = 0 
            batch_psnr = 0
            for i in range(ground_truth_image.shape[0]):
                contrast_ground_truth = ground_truth_image[i, contrast_index, :, :]
                contrast_estimated = estimated_image[i, contrast_index, :, :]
                contrast_ground_truth = contrast_ground_truth[None, None, :, :]
                contrast_estimated = contrast_estimated[None, None, :, :]


                max_val = max(contrast_ground_truth.max().item(), contrast_estimated.max().item())
                min_val = min(contrast_ground_truth.min().item(), contrast_estimated.min().item())
                batch_nmse = nmse(contrast_ground_truth, contrast_estimated)
                batch_ssim, ssim_image = ssim(contrast_ground_truth, contrast_estimated, return_full_image=True, data_range=(min_val, max_val))
                batch_psnr = psnr(contrast_ground_truth, contrast_estimated, mask)

                # remove mask points that would equal to 1 (possibly some estimated points
                # will be removed here but only if matches completely in the kernel)
                                                        
                batch_ssim = ssim_image[mask[i, contrast_index].unsqueeze(0).unsqueeze(0)].mean()

                #self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"metrics/{label}nmse_" + self.contrast_order[contrast_index], batch_nmse, sync_dist=True)
            self.log(f"metrics/{label}ssim_torch_" + self.contrast_order[contrast_index], batch_ssim, sync_dist=True)
            self.log(f"metrics/{label}psnr_" + self.contrast_order[contrast_index], batch_psnr, sync_dist=True)

            total_ssim += batch_ssim
            total_psnr += batch_psnr
            total_nmse += batch_nmse

        self.log(f'metrics/{label}_mean_ssim', total_ssim/len(self.contrast_order), on_epoch=True, sync_dist=True)
        self.log(f'metrics/{label}_mean_psnr', total_psnr/len(self.contrast_order), on_epoch=True, sync_dist=True)
        self.log(f'metrics/{label}_mean_nmse', total_nmse/len(self.contrast_order), on_epoch=True, sync_dist=True)
        return estimated_image

    def norm(self, image): 
        normed_image = image/image.amax((-1, -2), keepdim=True)
        return normed_image


    def plot_images(self, under_k, estimate_k, target, k_space, mask, mode='train'):
        with torch.no_grad():
            
            estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)
            image = root_sum_of_squares(ifft_2d_img(k_space), coil_dim=2)
            target = root_sum_of_squares(ifft_2d_img(target), coil_dim=2)

            estimated_image = estimated_image[0]/image[0].amax((-1, -2), keepdim=True)
            image = image[0]/image[0].amax((-1, -2), keepdim=True)
            target = target[0]/target[0].amax((-1, -2), keepdim=True)
            diff = (estimated_image - image).abs()*10
            k_space_scaled = k_space.abs()/(k_space.abs().max() / 50) 
            under_k = under_k.abs()/(under_k.abs().max() / 50)

            estimated_image = estimated_image.clamp(0, 1)
            image = image.clamp(0, 1)
            diff = diff.clamp(0, 1)
            under_k = under_k.clamp(0, 1)
            k_space_scaled = k_space_scaled.clamp(0, 1)
            wandb_logger = self.logger

            contrasts = estimated_image.shape[0]
            wandb_logger.log_image(mode + '/recon', np.split(estimated_image.unsqueeze(1).cpu().numpy(), contrasts, 0))
            wandb_logger.log_image(mode + '/fully_sampled', np.split(image.unsqueeze(1).cpu().numpy(), contrasts, 0))
            wandb_logger.log_image(mode + '/mask', np.split(mask[0, :, [0], :, :].cpu().numpy(), contrasts, 0))
            wandb_logger.log_image(mode + '/diff', np.split(diff.unsqueeze(1).cpu().numpy(), contrasts, 0))
