import torch
import numpy as np
from torch import optim

import pytorch_lightning as pl
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pytorch_lightning.loggers import WandbLogger

from ml_recon.utils import ifft_2d_img, root_sum_of_squares
from ml_recon.losses import L1L2Loss, L1ImageGradLoss
from ml_recon.models.MultiContrastVarNet import MultiContrastVarNet, VarnetConfig
from ml_recon.pl_modules.pl_ReconModel import plReconModel

from typing import Literal, Optional
from functools import partial

# define the LightningModule
class pl_VarNet(plReconModel):
    def __init__(
            self, 
            config: VarnetConfig = VarnetConfig(['t1']),
            image_loss_function: Optional[str] = '',
            image_loss_scaling: float = 1,
            k_loss_function: Optional[str] = 'norml1l2',
            k_loss_scaling: float = 1.0,
            norm_all_k: bool = False,
            lr: float = 1e-3,
            is_supervised: bool = False
            ):

        self.config = config

        super().__init__(config.contrast_order)

        self.save_hyperparameters()

        # reconstruction model
        self.model = MultiContrastVarNet(config)
        
        # set learning rate
        self.lr = lr
        self.is_supervised = is_supervised

        # Set loss functions
        self.k_loss_func = self._set_k_loss_func(k_loss_function, norm_all_k)
        self.image_loss_func = self._set_image_loss_func(image_loss_function)
        self.k_loss_scaling = k_loss_scaling
        self.image_loss_scaling = image_loss_scaling


    def training_step(self, batch, batch_idx):
        """Training step for varnet

        Args:
            batch (dict): dictionary containing: 'input', 'target', 'fs_k_space', 'mask', 'loss_mask'
            batch_idx (int): current batch index

        Returns:
            float: loss value
        """
        undersampled = batch['undersampled']
        fully_sampled = batch['fs_k_space']
        mask = batch['mask']
        prediction = self.forward(undersampled, mask, fully_sampled) #fully sampled here used for zero filling mask
        target = fully_sampled * batch['loss_mask'] # loss mask is all ones in fully supervised case

        target_img = root_sum_of_squares(ifft_2d_img(target), coil_dim=2)
        estimated_img = root_sum_of_squares(ifft_2d_img(prediction), coil_dim=2)
        
        loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        if self.k_loss_func:
            loss += self.k_loss_func(target, prediction*batch['loss_mask']) * self.k_loss_scaling
            #loss = loss / loss_mask.sum()
        if self.image_loss_func: 
            loss += self.image_loss_func(target_img, estimated_img) * self.image_loss_scaling

        self.log('train/train_loss', loss, on_epoch=True, on_step=True, logger=True, sync_dist=True)
        if self.current_epoch % 1 == 0 and batch_idx == 0: 
            self.plot_example_images(batch, 'train')

        return loss

  

    def validation_step(self, batch, batch_idx):
        undersampled = batch['undersampled']
        fully_sampled = batch['fs_k_space']
        mask = batch['mask']
        prediction_lambda = self.forward(undersampled * mask, mask, fully_sampled) #fully sampled here used for zero filling mask
        target = fully_sampled * batch['loss_mask'] # loss mask is all ones in fully supervised case

        loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        if self.k_loss_func:
            loss += self.k_loss_func(target, prediction_lambda*batch['loss_mask'])


        inital_mask = batch['mask'] 
        if (batch['loss_mask'] == 0).any():
            inital_mask = inital_mask + batch['loss_mask']
        prediction_full = self.forward(undersampled, inital_mask, fully_sampled)
        ssim = self.calculate_ssim(batch['fs_k_space'], prediction_full)

        for contrast, ssim_contrast in zip(self.contrast_order, ssim):
            self.log(f'val/ssim_full_{contrast}', ssim_contrast, on_epoch=True, logger=True, sync_dist=True)
        self.log('val/val_loss', loss, on_epoch=True, logger=True, sync_dist=True)

        if self.current_epoch % 1 == 0 and batch_idx == 0: 
            self.plot_example_images(batch, 'val')
        return loss


    def test_step(self, batch, batch_index):
        mask = batch['mask']
        if (batch['loss_mask'] == 0).any(): # if loss mask has zeros (must be ssl loss mask)
            mask += batch['loss_mask'] # combine to get original sampliing mask

        estimate_k = self.forward(batch['undersampled'], mask, batch['fs_k_space'])
        return super().test_step((estimate_k, batch['fs_k_space']), batch_index)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        mask = batch['mask']
        if (batch['loss_mask'] == 0).any(): # if loss mask has zeros (must be ssl loss mask)
            mask += batch['loss_mask'] # combine to get original sampliing mask

        estimate_k = self.forward(batch['undersampled'], mask, batch['fs_k_space'])

        return super().on_test_batch_end(outputs, (estimate_k, batch['fs_k_space']), batch_idx, dataloader_idx)


    def forward(self, under_k, mask, fs_k_space): 
        zero_fill_mask = fs_k_space != 0
        estimate_k = self.model(under_k, mask)
        estimate_k = estimate_k * (1 - mask) + under_k
        return estimate_k * zero_fill_mask

    # optimizer configureation -> using adam w/ lr of 1e-3
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 6000, eta_min=1e-3) 
        return optimizer

    def plot_example_images(self, batch, mode='train'):
        with torch.no_grad():
            undersampled, mask, fs_k_space = batch['undersampled'], batch['mask'], batch['fs_k_space']
            
            # pass original data through model
            estimate_k = self.forward(undersampled, mask, fs_k_space)
            super().plot_images(estimate_k, fs_k_space, mask, mode) 

            sense_maps = self.model.sens_model(undersampled, mask)
            sense_maps = sense_maps[0, 0, :, :, :].unsqueeze(1).abs()

            # pass lambda set through model (if self supervised)
            target = undersampled * batch['loss_mask']
            undersampled = undersampled[0, :, [0], :, :].abs()**0.2
            target = target[0, :, [0], :, :].abs()**0.2
            
            if isinstance(self.logger, WandbLogger):
                wandb_logger = self.logger
                wandb_logger.log_image(mode + '/sense_maps', np.split(sense_maps.cpu().numpy()/sense_maps.max().item(), sense_maps.shape[0], 0))
                wandb_logger.log_image(mode + '/target', np.split(target.clamp(0, 1).cpu().numpy(),target.shape[0], 0))
                wandb_logger.log_image(mode + '/input', np.split(undersampled.clamp(0, 1).cpu().numpy(), undersampled.shape[0], 0))

    def _set_image_loss_func(self, image_loss_function):
        if image_loss_function == 'ssim':
            ssim_func = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(self.device)
            return lambda targ, pred: torch.tensor([1], device=self.device) - ssim_func(targ, pred)
        elif image_loss_function == 'l1':
            return torch.nn.L1Loss()
        elif image_loss_function == 'l1_grad':
            return L1ImageGradLoss(2)
        elif image_loss_function == 'l2':
            return torch.nn.MSELoss()
        else:
            print('No image space loss!!')
            return None
    
    def _set_k_loss_func(self, k_loss_function, norm_all_k):
        if k_loss_function == 'norml1l2':
            loss_func = L1L2Loss(norm_all_k)
        elif k_loss_function == 'l1':
            loss_func = torch.nn.L1Loss(reduction='sum')
        elif k_loss_function == 'l2':
            loss_func = torch.nn.MSELoss(reduction='sum')
        else:
            print('No k-space loss!!!')
            return None

        return lambda target, pred: loss_func(torch.view_as_real(target), torch.view_as_real(pred))
        

    def calculate_ssim(self, fully_sampled_k, estimate_target):
        """Calculates ssim between two MRI images, data range is set between each image

        Args:
            fully_sampled_k (torch.Tensor): fully sampled k-space
            estimate_target (torch.Tensor): estimated k-space

        Returns:
            torch.Tensor: float value of ssim averaged over batch and contrasts
        """
        ssim_func = structural_similarity_index_measure
        est_img = root_sum_of_squares(ifft_2d_img(estimate_target, axes=[-1, -2]), coil_dim=2)
        targ_img = root_sum_of_squares(ifft_2d_img(fully_sampled_k, axes=[-1, -2]), coil_dim=2)

        ssim_values = []
        for contrast in range(est_img.shape[1]):
            ssim = 0
            for i in range(est_img.shape[0]):
                image_max = targ_img[i, contrast, ...].max().item()
                ssim_val = ssim_func(
                    est_img[i, contrast, ...].unsqueeze(0).unsqueeze(0), 
                    targ_img[i, contrast, ...].unsqueeze(0).unsqueeze(0), 
                    data_range=(0, image_max)
                    )
                assert isinstance(ssim_val, torch.Tensor)
                ssim += ssim_val
            ssim /= est_img.shape[0]
            ssim_values.append(ssim)

        return ssim_values