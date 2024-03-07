import torch
import numpy as np
from torch import optim

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torchmetrics import StructuralSimilarityIndexMeasure

from ml_recon.losses import L1L2Loss
from ml_recon.models.varnet_mc import VarNet_mc
from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.utils.evaluate import nmse, ssim, psnr
from ml_recon.pl_modules.pl_model import plReconModel

from typing import Literal
from functools import partial

# define the LightningModule
class pl_VarNet(plReconModel):
    def __init__(
            self, 
            backbone: partial,
            contrast_order,
            num_cascades: int = 5, 
            sense_chans: int = 8,
            lr: float = 1e-3
            ):

        super().__init__(contrast_order)

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
        print(batch_idx)
        under, target = batch

        estimate_target = self.model(under, under != 0)

        loss = self.loss(target, estimate_target)
        self.log('train_loss', loss, on_epoch=True, on_step=True, logger=True)
        if batch_idx == 0: 
            self.plot_images((under, target), under != 0, 'train')

        return loss


    def validation_step(self, batch, batch_idx):
        under, target = batch
        estimate_target = self.model(under, under != 0)

        loss = self.loss(target, estimate_target)
        self.log('val_loss', loss, on_epoch=True, logger=True)
        if batch_idx == 0: 
            self.plot_images((under, target), under != 0 , 'val')
        return loss


    def forward(self, data, sampling_mask): 
        under_k, k_space = data
        print(under_k.type())
        print(k_space.type())
        return self.model(under_k, sampling_mask)

    # optimizer configureation -> using adam w/ lr of 1e-3
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 6000, eta_min=1e-3) 
        return [optimizer], [scheduler]

    def plot_images(self, batch, sampling_mask, mode='train'):
        #pass
        under_k, k_space = batch
        super().plot_images((under_k, k_space), sampling_mask, mode) 
        with torch.no_grad():
            tensorboard = self.logger.experiment
            sampling_mask = under_k != 0

            sense_maps = self.model.sens_model(under_k, under_k != 0)
            sense_maps = sense_maps[0, 0, :, :, :].unsqueeze(1).abs()
            masked_k = self.model.sens_model.mask(under_k, sampling_mask.expand_as(k_space))
            masked_k = masked_k[0, 0, [0], :, :].abs()/(masked_k[0, 0, [0], :, :].abs().max()/20)

            tensorboard.add_images(mode + '/sense_maps', sense_maps/sense_maps.max(), self.current_epoch)
            tensorboard.add_image(mode + '/masked_k', masked_k.clamp(0, 1), self.current_epoch)

            if isinstance(self.loggers, list): 
                wandb_logger = None
                for logger in self.loggers:
                    if isinstance(logger, WandbLogger):
                        wandb_logger = logger

                if wandb_logger:
                    assert isinstance(wandb_logger, WandbLogger)
                    wandb_logger.log_image(mode + '/sense_maps', np.split(sense_maps.cpu().numpy()/sense_maps.max().item(), sense_maps.shape[0], 0))
                    wandb_logger.log_image(mode + '/masked_k', [masked_k.clamp(0, 1).cpu().numpy()])
