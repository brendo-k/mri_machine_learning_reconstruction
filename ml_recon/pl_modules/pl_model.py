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
        under, k_space = batch

        estimate_k = self(batch, under != 0)

        #loss = self.loss(estimate_k, k_space)
        ssim_loss = StructuralSimilarityIndexMeasure().to(self.device)
        estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)
        ground_truth_image = root_sum_of_squares(ifft_2d_img(k_space), coil_dim=2) 
        total_ssim = 0
        total_psnr = 0
        total_nmse = 0

        output_mask = (ground_truth_image > 0.01)
        estimated_image *= output_mask
        ground_truth_image *= output_mask


        for contrast in range(len(self.contrast_order)):
            batch_nmse = nmse(ground_truth_image[:, [contrast], :, :], estimated_image[:, [contrast], :, :])
            batch_ssim = ssim(ground_truth_image[:, [contrast], :, :], estimated_image[:, [contrast], :, :], self.device)
            batch_ssim_torch = ssim_loss(estimated_image[:, [contrast], :, :], ground_truth_image[:, [contrast], :, :])
            batch_psnr = psnr(ground_truth_image[:, [contrast], :, :], estimated_image[:, [contrast], :, :])

            #self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log("nmse_" + self.contrast_order[contrast], batch_nmse, on_epoch=True, prog_bar=True, logger=True)
            self.log("ssim_" + self.contrast_order[contrast], batch_ssim, on_epoch=True, prog_bar=True, logger=True)
            self.log("ssim_torch_" + self.contrast_order[contrast], batch_ssim_torch, on_epoch=True, prog_bar=True, logger=True)
            self.log("psnr_" + self.contrast_order[contrast], batch_psnr, on_epoch=True, prog_bar=True, logger=True)

            total_ssim += batch_ssim
            total_psnr += batch_psnr
            total_nmse += batch_nmse

        self.log('mean_ssim', total_ssim/len(self.contrast_order), on_epoch=True)
        self.log('mean_psnr', total_psnr/len(self.contrast_order), on_epoch=True)
        self.log('mean_nmse', total_nmse/len(self.contrast_order), on_epoch=True)

        tensorboard_logger = None
        if isinstance(self.loggers, list): 
            for logger in self.loggers: 
                if isinstance(logger, TensorBoardLogger):
                    tensorboard_logger = logger
            if tensorboard_logger:
                tensorboard_logger.log_hyperparams(self.hparams, {
                   'mean_ssim': total_ssim/len(self.contrast_order),
                   'mean_psnr': total_psnr/len(self.contrast_order),
                   'mean_nmse': total_nmse/len(self.contrast_order),
                   })

    def plot_images(self, batch, sampling_mask, mode='train'):
        with torch.no_grad():
            under_k, k_space = batch

            estimate_k = self(batch, sampling_mask)
            
            estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)
            image = root_sum_of_squares(ifft_2d_img(k_space), coil_dim=2)

            estimated_image = estimated_image[0]/estimated_image[0].max()
            image = image[0]/image[0].max()
            tensorboard = self.logger.experiment


            k_space_scaled = k_space.abs()/(k_space.abs().max() / 20) 
            under_k = under_k.abs()/(under_k.abs().max() / 20)
            diff = (estimated_image - image).abs()

            tensorboard.add_images(mode + '/recon', estimated_image.unsqueeze(1), self.current_epoch)
            tensorboard.add_images(mode + '/target', image.unsqueeze(1), self.current_epoch)
            tensorboard.add_images(mode + '/diff', diff.unsqueeze(1), self.current_epoch)
            tensorboard.add_image(mode + '/k', k_space_scaled[0, 0, [0]].clamp(0, 1), self.current_epoch)
            tensorboard.add_image(mode + '/under_k', under_k[0, 0, [0]].clamp(0, 1), self.current_epoch)
            tensorboard.add_images(mode + '/mask', sampling_mask[0, :, [0], :, :], self.current_epoch)
            if isinstance(self.loggers, list): 

                wandb_logger = None
                for logger in self.loggers: 
                    if isinstance(logger, WandbLogger):
                        wandb_logger = logger
                if wandb_logger:
                    assert isinstance(wandb_logger, WandbLogger)
                    contrasts = estimated_image.shape[0]
                    wandb_logger.log_image(mode + '/recon', np.split(estimated_image.unsqueeze(1).cpu().numpy(), contrasts, 0))
                    wandb_logger.log_image(mode + '/target', np.split(image.unsqueeze(1).cpu().numpy(), contrasts, 0))
                    wandb_logger.log_image(mode + '/k', [k_space_scaled[0, 0, [0]].clamp(0, 1).cpu().numpy()])
                    wandb_logger.log_image(mode + '/under_k', [under_k[0, 0, [0]].clamp(0, 1).cpu().numpy()])
                    wandb_logger.log_image(mode + '/mask', np.split(sampling_mask[0, :, [0], :, :].cpu().numpy(), contrasts, 0))
                    wandb_logger.log_image(mode + '/diff', np.split(diff.unsqueeze(1).cpu().numpy(), contrasts, 0))
