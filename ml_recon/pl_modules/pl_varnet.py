import torch
import numpy as np
from torch import optim

import pytorch_lightning as pl
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pytorch_lightning.loggers import WandbLogger

from ml_recon.utils import ifft_2d_img, root_sum_of_squares
from ml_recon.losses import L1L2Loss, L1ImageGradLoss
from ml_recon.models.varnet_mc import VarNet_mc
from ml_recon.pl_modules.pl_ReconModel import plReconModel
from ml_recon.models import Unet
from ml_recon.models import ResNet
from ml_recon.models import UnetR
from ml_recon.models import SingleEncoderJointDecoder
from monai.networks.nets.swin_unetr import SwinUNETR

from typing import Literal, Optional
from functools import partial
from dataclasses import dataclass

@dataclass
class VarnetConfig:
    contrast_order: list
    model_name: str = 'unet'
    cascades: int = 5
    sense_chans: int = 8
    lr: float = 1e-3
    channels: int = 18
    image_loss_function: Optional[str] = ''
    image_loss_scaling: float = 0
    k_loss_function: Optional[str] = 'norml1l2'
    k_loss_scaling: float = 0
    norm_all_k: bool = False
    split_contrast_by_phase: bool = False

# define the LightningModule
class pl_VarNet(plReconModel):
    def __init__(
            self, 
            config: VarnetConfig = VarnetConfig(['t1'])
            ):

        self.config = config

        super().__init__(config.contrast_order)

        self.save_hyperparameters()

        # get the VarNet backbone (refinement module)
        backbone = self._create_backbone()
       
        # reconstruction model
        self.model = VarNet_mc(
            backbone,
            contrasts=len(self.contrast_order),
            num_cascades=self.config.cascades, 
            sens_chans=self.config.cascades, 
            split_complex_by_phase = config.split_contrast_by_phase
        )
        
        # set learning rate
        self.lr = self.config.lr

        # Set loss functions
        self.k_loss_func = self._set_k_loss_func()
        self.image_loss_func = self._set_image_loss_func()



    def training_step(self, batch, batch_idx):
        """Training step for varnet

        Args:
            batch (dict): dictionary containing: 'input', 'target', 'fs_k_space', 'mask', 'loss_mask'
            batch_idx (int): current batch index

        Returns:
            float: loss value
        """

        prediction = self.forward(batch)

        target_img = root_sum_of_squares(ifft_2d_img(batch['target']), coil_dim=1)
        estimated_img = root_sum_of_squares(ifft_2d_img(prediction), coil_dim=1)
        
        loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        if self.k_loss_func:
            loss += self.k_loss_func(batch['target'], prediction*batch['loss_mask'])
        if self.image_loss_func: 
            loss += self.image_loss_func(target_img, estimated_img) * self.image_space_scaling

        self.log('train/train_loss', loss, on_epoch=True, on_step=True, logger=True, sync_dist=True)
        if self.current_epoch % 10 == 0 and batch_idx == 0: 
            self.plot_example_images(batch, 'train')

        return loss


    def validation_step(self, batch, batch_idx):
        prediction = self.forward(batch)

        target_img = root_sum_of_squares(ifft_2d_img(batch['target']), coil_dim=1)
        estimated_img = root_sum_of_squares(ifft_2d_img(prediction), coil_dim=1)

        loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        if self.k_loss_func:
            loss += self.k_loss_func(batch['target'], prediction*batch['loss_mask'])
        if self.image_loss_func: 
            loss += self.image_loss_func(target_img, estimated_img) * self.image_space_scaling

        ssim = self.calculate_ssim(batch['fs_k_space'], prediction)

        for contrast, ssim_contrast in zip(self.contrast_order, ssim):
            self.log(f'val/ssim_{contrast}', ssim_contrast, on_epoch=True, logger=True, sync_dist=True)
        self.log('val/val_loss', loss, on_epoch=True, logger=True, sync_dist=True)

        if self.current_epoch % 10 == 0 and batch_idx == 0: 
            self.plot_example_images(batch, 'val')
        return loss


    def test_step(self, batch, batch_index):
        estimated_target = self.forward(batch)
        return super().test_step((estimated_target, batch['fs_k_space']), batch_index)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):

        estimate_k = self.forward(batch)

        return super().on_test_batch_end(outputs, (estimate_k, batch['fs_k_space']), batch_idx, dataloader_idx)


    def forward(self, data): 
        under_k, mask, fs_k_space = data['input'], data['mask'], data['fs_k_space']
        zero_fill_mask = fs_k_space != 0
        estimate_k = self.model(under_k, mask)
        estimate_k = estimate_k * ~mask + under_k
        return estimate_k * zero_fill_mask

    # optimizer configureation -> using adam w/ lr of 1e-3
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 6000, eta_min=1e-3) 
        return optimizer

    def plot_example_images(self, batch, mode='train'):
        #pass
        under_k = batch['input']
        with torch.no_grad():
            estimate_k = self.forward(batch)
            estimate_k = estimate_k * (batch['input'] == 0) + batch['input']
            super().plot_images(under_k, estimate_k, batch['target'], batch['fs_k_space'], batch['mask'], mode) 

            sense_maps = self.model.sens_model(under_k, batch['mask'])
            sense_maps = sense_maps[0, 0, :, :, :].unsqueeze(1).abs()
            masked_k = self.model.sens_model.mask(under_k, batch['mask'])
            masked_k = masked_k[0, :, [0], :, :].abs()**0.2

            input = batch['input'][0, :, [0], :, :].abs()**0.2
            target = batch['target'][0, :, [0], :, :].abs()**0.2
            if isinstance(self.logger, WandbLogger):
                wandb_logger = self.logger
                wandb_logger.log_image(mode + '/target', np.split(target.clamp(0, 1).cpu().numpy(),target.shape[0], 0))
                wandb_logger.log_image(mode + '/input', np.split(input.clamp(0, 1).cpu().numpy(), input.shape[0], 0))

    def _set_image_loss_func(self):
        if self.config.image_loss_function == 'ssim':
            ssim_func = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(self.device)
            return lambda targ, pred: torch.tensor([1], device=self.device) - ssim_func(targ, pred)
        elif self.config.image_loss_function == 'l1':
            return torch.nn.L1Loss()
        elif self.config.image_loss_function == 'l1_grad':
            return L1ImageGradLoss(2)
        elif self.config.image_loss_function == 'l2':
            return torch.nn.MSELoss()
        else:
            print('No image space loss!!')
            return None
    
    def _set_k_loss_func(self):
        if self.config.k_loss_function == 'norml1l2':
            loss_func = L1L2Loss(self.config.norm_all_k)
        elif self.config.k_loss_function == 'l1':
            loss_func = torch.nn.L1Loss()
        elif self.config.k_loss_function == 'l2':
            loss_func = torch.nn.MSELoss()
        else:
            print('No k-space loss!!!')
            return None

        return lambda target, pred: loss_func(torch.view_as_real(target), torch.view_as_real(pred))
        


        
    def _create_backbone(self):
        contrast_len = 2 * len(self.config.contrast_order)
        backbone_params = {
            'in_chan': contrast_len,
            'out_chan': contrast_len,
            'chans': self.config.channels
        }
        
        if self.config.model_name == 'unet':
            return self._create_unet(backbone_params)
        elif self.config.model_name == 'resnet':
            return self._create_resnet(backbone_params)
        elif self.config.model_name == 'se_jd':
            return self._create_se_jd(backbone_params)
        elif self.config.model_name == 'unetr':
            return self._create_unetr(backbone_params)
        elif self.config.model_name == 'swin_unetr':
            return self._create_swin_unetr(backbone_params)
        else:
            raise ValueError(f"Model {self.config.model_name} not found!")
        
        
    def _create_unet(self, backbone_params):
        return partial(Unet, **backbone_params)
    
    def _create_resnet(self, backbone_params):
        return partial(ResNet, **backbone_params, itterations=15)
    
    def _create_se_jd(self, backbone_params):
        return partial(
            SingleEncoderJointDecoder,
            in_chan=backbone_params['in_chan'],
            encoder_chan=16, 
            encoder_depth=4, 
            decoder_chan=backbone_params['chans'], 
            decoder_depth=4
        )
    
    def _create_unetr(self, backbone_params):
        return partial(
            UnetR,
            in_chan=backbone_params['in_chan'],
            out_chan=backbone_params['out_chan'],
            hidden_size=backbone_params['chans'],
            img_size=128
        )

    def _create_swin_unetr(self, backbone_params):
        return partial(
            SwinUNETR,
            in_channels=backbone_params['in_chan'],
            out_channels=backbone_params['out_chan'],
            feature_size=backbone_params['chans'],
            img_size=128,
            spatial_dims=2
        )


    def calculate_ssim(self, fully_sampled_k, estimate_target):
        """Calculates ssim between two MRI images, data range is set between each image

        Args:
            fully_sampled_k (torch.Tensor): fully sampled k-space
            estimate_target (torch.Tensor): estimated k-space

        Returns:
            torch.Tensor: float value of ssim averaged over batch and contrasts
        """
        ssim_func = structural_similarity_index_measure
        est_img = root_sum_of_squares(ifft_2d_img(estimate_target, axes=[-1, -2]), coil_dim=1)
        targ_img = root_sum_of_squares(ifft_2d_img(fully_sampled_k, axes=[-1, -2]), coil_dim=1)

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