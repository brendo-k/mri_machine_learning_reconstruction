import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torchmetrics import StructuralSimilarityIndexMeasure

from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.utils.evaluate import nmse, ssim, psnr

# define the LightningModule
class plReconModel(pl.LightningModule):
    def __init__(
            self, 
            contrast_order,
            ):

        super().__init__()
        self.contrast_order = contrast_order

    def test_step(self, batch, _):
        estimate_k, k_space = batch

        #loss = self.loss(estimate_k, k_space)
        ssim_loss = StructuralSimilarityIndexMeasure().to(self.device)
        estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)
        ground_truth_image = root_sum_of_squares(ifft_2d_img(k_space), coil_dim=2) 
        total_ssim = 0
        total_psnr = 0
        total_nmse = 0
        mask = ground_truth_image > 0.0015
        estimated_image *= mask
        ground_truth_image *= mask


        wandb_logger = self.logger
        contrasts = estimated_image.shape[1]
        for i in range(estimated_image.shape[0]):
            wandb_logger.log_image('test' + '/recon', np.split(estimated_image[i].unsqueeze(1).cpu().numpy(), contrasts, 0))
            wandb_logger.log_image('test' + '/target', np.split(ground_truth_image[i].unsqueeze(1).cpu().numpy(), contrasts, 0))

        for contrast in range(len(self.contrast_order)):
            batch_nmse = nmse(ground_truth_image[:, [contrast], :, :], estimated_image[:, [contrast], :, :])
            batch_ssim_torch = ssim_loss(estimated_image[:, [contrast], :, :], ground_truth_image[:, [contrast], :, :])
            batch_psnr = psnr(ground_truth_image[:, [contrast], :, :], estimated_image[:, [contrast], :, :])

            #self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log("metrics/nmse_" + self.contrast_order[contrast], batch_nmse, on_epoch=True, prog_bar=True, logger=True)
            self.log("metrics/ssim_torch_" + self.contrast_order[contrast], batch_ssim_torch, on_epoch=True, prog_bar=True, logger=True)
            self.log("metrics/psnr_" + self.contrast_order[contrast], batch_psnr, on_epoch=True, prog_bar=True, logger=True)

            total_ssim += batch_ssim_torch
            total_psnr += batch_psnr
            total_nmse += batch_nmse

        self.log('metrics/mean_ssim', total_ssim/len(self.contrast_order), on_epoch=True)
        self.log('metrics/mean_psnr', total_psnr/len(self.contrast_order), on_epoch=True)
        self.log('metrics/mean_nmse', total_nmse/len(self.contrast_order), on_epoch=True)


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
            wandb_logger.log_image(mode + '/target', np.split(target.unsqueeze(1).cpu().numpy(), contrasts, 0))
            wandb_logger.log_image(mode + '/k', [k_space_scaled[0, 0, [0]].clamp(0, 1).cpu().numpy()])
            wandb_logger.log_image(mode + '/under_k', [under_k[0, 0, [0]].clamp(0, 1).cpu().numpy()])
            wandb_logger.log_image(mode + '/mask', np.split(mask[0, :, [0], :, :].cpu().numpy(), contrasts, 0))
            wandb_logger.log_image(mode + '/diff', np.split(diff.unsqueeze(1).cpu().numpy(), contrasts, 0))
