import torch
import numpy as np
from torch import optim

import pytorch_lightning as pl
from torchmetrics.image import StructuralSimilarityIndexMeasure

from ml_recon.utils import ifft_2d_img, root_sum_of_squares
from ml_recon.losses import L1L2Loss
from ml_recon.models.varnet_mc import VarNet_mc
from ml_recon.pl_modules.pl_model import plReconModel
from ml_recon.models import Unet
from ml_recon.models import ResNet

from typing import Literal
from functools import partial

# define the LightningModule
class pl_VarNet(plReconModel):
    def __init__(
            self, 
            contrast_order,
            model_name: str = 'unet',
            num_cascades: int = 5, 
            sense_chans: int = 8,
            lr: float = 1e-3,
            chans = 18, 
            ):

        super().__init__(contrast_order)

        self.save_hyperparameters()
        if model_name == 'unet':
            backbone = partial(Unet, in_chan=2*len(contrast_order), out_chan=2*len(contrast_order), chans=chans)
        elif model_name == 'resnet':
            backbone = partial(ResNet, in_chan=2*len(contrast_order), out_chan=2*len(contrast_order), chans=chans)
        else:
            raise ValueError(f'{model_name} not found!')

        self.model = VarNet_mc(
            backbone,
            num_cascades, 
            sense_chans,
        )
        self.lr = lr
        self.contrast_order = contrast_order
        self.loss = lambda target, prediction: L1L2Loss(torch.view_as_real(target), torch.view_as_real(prediction))

    def training_step(self, batch, batch_idx):

        estimate_target = self(batch)
        estimate_target = estimate_target * (batch['input'] == 0) + batch['input']

        loss = self.loss(batch['target'], estimate_target*batch['loss_mask'])

        self.log('train/train_loss', loss, on_epoch=True, on_step=True, logger=True)

        if batch_idx == 0: 
            self.plot_images(batch, 'train')

        return loss


    def validation_step(self, batch, batch_idx):
        estimate_target = self.forward(batch)
        estimate_target = estimate_target * (batch['input'] == 0) + batch['input']

        loss = self.loss(batch['target'], estimate_target*batch['loss_mask'])
        self.log('val/val_loss', loss, on_epoch=True, logger=True)

        ssim_func = StructuralSimilarityIndexMeasure().to(self.device)
        est_img = root_sum_of_squares(ifft_2d_img(estimate_target, axes=[-1, -2]), coil_dim=1)
        targ_img = root_sum_of_squares(ifft_2d_img(batch['fs_k_space'], axes=[-1, -2]), coil_dim=1)
        ssim = ssim_func(est_img, targ_img)

        self.log('val/ssim', ssim, on_epoch=True, logger=True)
        if batch_idx == 0: 
            self.plot_images(batch, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        estimated_target = self.forward(batch)
        estimated_target = estimated_target * (batch['input'] == 0) + batch['input']
        super().test_step((estimated_target, batch['fs_k_space']), None)


    def forward(self, data): 
        under_k, mask = data['input'], data['mask']
        estimate_k = self.model(under_k, mask)
        return estimate_k

    # optimizer configureation -> using adam w/ lr of 1e-3
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 6000, eta_min=1e-3) 
        return optimizer

    def plot_images(self, batch, mode='train'):
        #pass
        under_k = batch['input']
        with torch.no_grad():
            estimate_k = self(batch)
            estimate_k = estimate_k * (batch['input'] == 0) + batch['input']
            super().plot_images(under_k, estimate_k, batch['target'], batch['fs_k_space'], batch['mask'], mode) 
            sampling_mask = under_k != 0

            sense_maps = self.model.sens_model(under_k, under_k != 0)
            sense_maps = sense_maps[0, 0, :, :, :].unsqueeze(1).abs()
            masked_k = self.model.sens_model.mask(under_k, sampling_mask.expand_as(under_k))
            masked_k = masked_k[0, 0, [0], :, :].abs()/(masked_k[0, 0, [0], :, :].abs().max()/20)

            wandb_logger = self.logger
            wandb_logger.log_image(mode + '/sense_maps', np.split(sense_maps.cpu().numpy()/sense_maps.max().item(), sense_maps.shape[0], 0))
            wandb_logger.log_image(mode + '/masked_k', [masked_k.clamp(0, 1).cpu().numpy()])
