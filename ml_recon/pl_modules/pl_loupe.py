import torch.nn as nn
import torch
import pytorch_lightning as L 
from pytorch_lightning.loggers import TensorBoardLogger
import einops

from ml_recon.models import VarNet_mc, Unet
from functools import partial
from ml_recon.utils import ifft_2d_img, complex_to_real, root_sum_of_squares
from ml_recon.losses import L1L2Loss
from ml_recon.utils.evaluate import ssim, psnr, nmse
from torchmetrics import StructuralSimilarityIndexMeasure

class LOUPE(L.LightningModule):
    def __init__(
            self, 
            recon_model,
            image_size, 
            R, 
            contrast_order, 
            center_region = 5,
            prob_method='loupe', 
            mask_method='all',
            lr=1e-2
            ):
        super().__init__()
        self.save_hyperparameters(ignore='recon_model')
        self.image_size = image_size
        self.contrast_order = contrast_order
        self.R = R
        self.recon_model = recon_model
        self.mask_method = mask_method
        self.lr = lr
        self.center_region = center_region

        self.sigmoid_slope_1 = 5
        self.sigmoid_slope_2 = 50
        self.prob_method = prob_method

        if prob_method == 'loupe':
            O = torch.rand(image_size)*(1 - 2e-2) + 1e-2
            self.sampling_weights = nn.Parameter(-torch.log(1/ O - 1) / self.sigmoid_slope_1)
        elif prob_method == 'gubmel':
            self.sampling_weights = nn.Parameter(torch.rand(image_size))
        elif prob_method == 'line_loupe':
            O = torch.rand(image_size[0], image_size[2])*(1 - 2e-2) + 1e-2
            self.sampling_weights = nn.Parameter(-torch.log(1/ O - 1) / self.sigmoid_slope_1)


    def training_step(self, batch, batch_idx):
        under, k_space = batch
        sampling_mask, under_k, estimate_k = self.pass_through_model(k_space)

        loss = L1L2Loss(torch.view_as_real(estimate_k), torch.view_as_real(k_space))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.plot_images(k_space, 'train') 
        return loss



    def validation_step(self, batch, batch_idx):
        under, k_space = batch
        sampling_mask, under_k, estimate_k = self.pass_through_model(k_space)

        loss = L1L2Loss(torch.view_as_real(estimate_k), torch.view_as_real(k_space))

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.plot_images(k_space, 'val') 

    def test_step(self, batch, batch_idx):
        under, k_space = batch
        ssim_loss = StructuralSimilarityIndexMeasure().to(self.device)

        sampling_mask, under_k, estimate_k = self.pass_through_model(k_space, deterministic=True)

        loss = L1L2Loss(torch.view_as_real(estimate_k), torch.view_as_real(k_space))

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

            self.log("metrics/test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log("metrics/nmse_" + self.contrast_order[contrast], batch_nmse, on_epoch=True, prog_bar=True, logger=True)
            self.log("metrics/ssim_" + self.contrast_order[contrast], batch_ssim, on_epoch=True, prog_bar=True, logger=True)
            self.log("metrics/ssim_torch_" + self.contrast_order[contrast], batch_ssim_torch, on_epoch=True, prog_bar=True, logger=True)
            self.log("metrics/psnr_" + self.contrast_order[contrast], batch_psnr, on_epoch=True, prog_bar=True, logger=True)

            total_ssim += batch_ssim
            total_psnr += batch_psnr
            total_nmse += batch_nmse

        self.log('metris/mean_ssim', total_ssim/len(self.contrast_order), on_epoch=True, logger=True, prog_bar=True)
        self.log('metris/mean_psnr', total_psnr/len(self.contrast_order), on_epoch=True, logger=True, prog_bar=True)
        self.log('metris/mean_nmse', total_nmse/len(self.contrast_order), on_epoch=True, logger=True, prog_bar=True)

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.log_hyperparams(self.save_hyperparameters(), {
               'mean_ssim': total_ssim/len(self.contrast_order),
               'mean_psnr': total_psnr/len(self.contrast_order),
               'mean_nmse': total_nmse/len(self.contrast_order),
               })
        

        if batch_idx == 0 and isinstance(self.logger, TensorBoardLogger):
            self.plot_images(k_space, 'test') 


    def norm_prob(self, probability):
        if self.prob_method == 'loupe' or self.prob_method == 'gumbel':
            mean_prob = torch.mean(probability, dim=[-1, -2]) 
            center = [probability.shape[-1]//2, probability.shape[-2]//2]
            
            alpha = 1/self.R 
            probability = alpha/mean_prob[:, None, None] * probability
            if self.mask_method == 'all':
                probability[:, center[0]-self.center_region//2:center[0]+self.center_region//2, center[1]-self.center_region//2:center[1]+self.center_region//2] = probability[:, center[0]-10:center[0]+10, center[1]-10:center[1]+10] + 2
            elif self.mask_method == 'first':
                probability[0, center[0]-10:center[0]+10, center[1]-10:center[1]+10] = probability[0, center[0]-10:center[0]+10, center[1]-10:center[1]+10] + 2
        elif self.prob_method == 'line_loupe':
            mean_prob = torch.mean(probability, dim=[-1]) 
            center = probability.shape[-1]//2
            
            alpha = 1/self.R 
            probability = alpha/mean_prob[:, None] * probability
            if self.mask_method == 'all':
                probability[:, center-10:center+10] = probability[:, center-10:center+10] + 2
            elif self.mask_method == 'first':
                probability[0, center-10:center+10] = probability[0, center-10:center+10] + 2

        return probability



    def pass_through_model(self, k_space, deterministic=False):
        # Calculate probability and normalize

        if 'loupe' in self.prob_method:
            probability = torch.sigmoid(self.sampling_weights * self.sigmoid_slope_1)
            assert probability.min() >= 0, f'Probability should be greater than 1 but found {probability.min()}'
            assert probability.max() <= 1, f'Probability should be less than 1 but found {probability.max()}'
        elif self.prob_method == 'gumbel': 
            probability = self.sampling_weights
        else:
            raise TypeError('prob_method should be loupe or gumbel')


        norm_probability = self.norm_prob(probability)

        if self.prob_method == 'loupe':
            activation = norm_probability - torch.rand((k_space.shape[0],) + self.image_size, device=self.device)
            # slope increase as epochs continue
            if self.current_epoch > 50:
                self.sigmoid_slope_2 += 2
            # Generate sampling mask
        elif self.prob_method == 'gumbel':
            activation = torch.log(norm_probability) + self.sample_gumbel((k_space.shape[0],) + self.image_size)
        elif self.prob_method == 'line_loupe':
            activation = norm_probability - torch.rand((k_space.shape[0], self.image_size[0], self.image_size[2]), device=self.device)
            activation = einops.repeat(activation, 'b c w -> b c h w', h=self.image_size[-1])
        else:
            raise ValueError('No sampling method!')

        if deterministic:
            sampling_mask = (activation > 0).to(torch.float)
        else:
            sampling_mask = torch.sigmoid(activation * self.sigmoid_slope_2)

    
        assert not torch.isnan(sampling_mask).any()
        # Ensure sampling mask values are within [0, 1]
        assert sampling_mask.min() >= 0 and sampling_mask.max() <= 1
    
        # Apply the sampling mask to k_space
        sampling_mask = sampling_mask.unsqueeze(2).expand(-1, -1, k_space.shape[2], -1, -1)
        
        assert sampling_mask.shape == k_space.shape, 'Sampling mask and k_space should have the same shape!'

        under_k = k_space * sampling_mask
    
        # Assert the pattern of zeros in the under-sampled k-space
        assert ((under_k[0, 0, 0, :, :] == 0) == (under_k[0, 0, 1, :, :] == 0)).all()
    
        # Estimate k-space using the model
        estimate_k = self.recon_model(under_k, sampling_mask)
    
        return sampling_mask, under_k, estimate_k


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 6000, eta_min=1e-2) 
        return [optimizer], [scheduler]

    def plot_images(self, k_space, mode='train'):
        with torch.no_grad():
            sampling_mask, under_k, estimate_k = self.pass_through_model(k_space)

            recon = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)
            recon = recon[0]/recon[0].max()

            image = root_sum_of_squares(ifft_2d_img(k_space), coil_dim=2)
            image = image[0]/image[0].max()
            tensorboard = self.logger.experiment

            sense_maps = self.recon_model.model.sens_model(under_k, sampling_mask.expand_as(k_space))
            sense_maps = sense_maps[0, 0, :, :, :].unsqueeze(1).abs()
            k_space_scaled = k_space.abs()/(k_space.abs().max() / 20) 
            under_k = under_k.abs()/(under_k.abs().max() / 20)

            tensorboard.add_images(mode + '/sense_maps', sense_maps/sense_maps.max(), self.current_epoch)
            tensorboard.add_images(mode + '/recon', recon.unsqueeze(1), self.current_epoch)
            tensorboard.add_images(mode + '/target', image.unsqueeze(1), self.current_epoch)
            tensorboard.add_images(mode + '/diff', (recon.unsqueeze(1)-image.unsqueeze(1)).abs(), self.current_epoch)
            tensorboard.add_image(mode + '/k', k_space_scaled[0, 0, [0]].clamp(0, 1), self.current_epoch)
            tensorboard.add_image(mode + '/under_k', under_k[0, 0, [0]].clamp(0, 1), self.current_epoch)
            tensorboard.add_images(mode + '/mask', sampling_mask[0, :, [0], :, :], self.current_epoch)
            if self.sampling_weights.ndim == 3:
                tensorboard.add_images(mode + '/sample_weights', self.sampling_weights.unsqueeze(1)/self.sampling_weights.max(), self.current_epoch)
            if self.sampling_weights.ndim == 2:
                weights = einops.repeat(self.sampling_weights, 'c h -> c 1 h w', w=self.sampling_weights.shape[1])
                tensorboard.add_images(mode + '/sample_weights', weights/weights.max(), self.current_epoch)


    def sample_gumbel(self, size, eps=1e-20): 
        U = torch.rand(size, device=self.device)
        return -torch.log(-torch.log(U + eps) + eps)
