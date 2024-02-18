import torch
from torch import optim

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import StructuralSimilarityIndexMeasure

from ml_recon.losses import L1L2Loss
from ml_recon.models.varnet_mc import VarNet_mc
from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.utils.evaluate import nmse, ssim, psnr

from typing import Literal
from functools import partial

# define the LightningModule
class pl_VarNet(pl.LightningModule):
    def __init__(
            self, 
            backbone: partial,
            contrast_order,
            num_cascades: int = 5, 
            sense_chans: int = 8,
            lr: float = 1e-3
            ):

        super().__init__()
        self.save_hyperparameters()
        self.model = VarNet_mc(
            backbone,
            num_cascades, 
            sense_chans,
        )
        self.lr = lr
        self.contrast_order = contrast_order
        self.loss = lambda target, prediction: L1L2Loss(torch.view_as_real(target), torch.view_as_real(prediction))

    def training_step(self, batch, batch_idx):
        under, target = batch

        estimate_target = self.model(under, under != 0)

        loss = self.loss(target, estimate_target)
        self.log('training_loss', loss, on_epoch=True, on_step=True)
        if batch_idx == 0: 
            self.plot_images(under, target, 'train')

            tensorboard = self.logger.experiment
            recon = root_sum_of_squares(ifft_2d_img(under), coil_dim=2)
            recon = recon[0]/recon[0].max()

            image = root_sum_of_squares(ifft_2d_img(target), coil_dim=2)
            image = image[0]/image[0].max()

            tensorboard.add_images('train' + '/target', image.unsqueeze(1), self.current_epoch)
            tensorboard.add_images('train' + '/estimate_target', recon.unsqueeze(1), self.current_epoch)
        return loss


    def validation_step(self, batch, batch_idx):
        under, target = batch
        estimate_target = self.model(under, under != 0)

        loss = self.loss(target, estimate_target)
        self.log('val_loss', loss, on_epoch=True)
        if batch_idx == 0: 
            self.plot_images(under, target, 'val')
        return loss

    def test_step(self, batch, _):
        under, k_space = batch
        ssim_loss = StructuralSimilarityIndexMeasure().to(self.device)

        estimate_k = self.model(under, under != 0)

        loss = self.loss(estimate_k, k_space)

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

            self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
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

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.log_hyperparams(self.save_hyperparameters(), {
               'mean_ssim': total_ssim/len(self.contrast_order),
               'mean_psnr': total_psnr/len(self.contrast_order),
               'mean_nmse': total_nmse/len(self.contrast_order),
               })

    def forward(self, data, mask): 
        return self.model(data, mask)

    # optimizer configureation -> using adam w/ lr of 1e-3
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 6000, eta_min=1e-3) 
        return [optimizer], [scheduler]

    def plot_images(self, under_k, k_space, mode='train'):
        with torch.no_grad():
            estimate_k = self.model(under_k, under_k != 0)

            recon = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)
            recon = recon[0]/recon[0].max()

            image = root_sum_of_squares(ifft_2d_img(k_space), coil_dim=2)
            image = image[0]/image[0].max()
            tensorboard = self.logger.experiment

            sampling_mask = under_k != 0

            sense_maps = self.model.sens_model(under_k, under_k != 0)
            sense_maps = sense_maps[0, 0, :, :, :].unsqueeze(1).abs()
            masked_k = self.model.sens_model.mask(under_k, sampling_mask.expand_as(k_space))
            masked_k = masked_k[0, 0, [0], :, :].abs()/(masked_k[0, 0, [0], :, :].abs().max()/20)
            k_space_scaled = k_space.abs()/(k_space.abs().max() / 20) 
            under_k = under_k.abs()/(under_k.abs().max() / 20)

            tensorboard.add_images(mode + '/sense_maps', sense_maps/sense_maps.max(), self.current_epoch)
            tensorboard.add_image(mode + '/masked_k', masked_k.clamp(0, 1), self.current_epoch)
            tensorboard.add_images(mode + '/recon', recon.unsqueeze(1), self.current_epoch)
            tensorboard.add_images(mode + '/target', image.unsqueeze(1), self.current_epoch)
            tensorboard.add_image(mode + '/k', k_space_scaled[0, 0, [0]].clamp(0, 1), self.current_epoch)
            tensorboard.add_image(mode + '/under_k', under_k[0, 0, [0]].clamp(0, 1), self.current_epoch)
            tensorboard.add_images(mode + '/mask', sampling_mask[0, :, [0], :, :], self.current_epoch)

