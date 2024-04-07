import torch.nn as nn
import numpy as np
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
import einops

from ml_recon.losses import L1L2Loss
from ml_recon.dataset.undersample import scale_pdf, gen_pdf_bern, gen_pdf_columns
from ml_recon.pl_modules.pl_model import plReconModel
from ml_recon.utils.evaluate import nmse, ssim, psnr
from ml_recon.utils import ifft_2d_img, root_sum_of_squares

class LOUPE(plReconModel):
    def __init__(
            self, 
            recon_model,
            image_size, 
            R: float, 
            contrast_order, 
            center_region = 10,
            prob_method='loupe', 
            mask_method='all',
            sigmoid_slope1 = 5,
            sigmoid_slope2 = 200,
            lr=1e-2,
            warm_start:bool = False, 
            lambda_param = 0,
            fd_param = 0,
            learn_R = False,
            ):
        super().__init__(contrast_order=contrast_order)
        self.save_hyperparameters(ignore='recon_model')
        self.image_size = image_size
        self.contrast_order = contrast_order
        self.R = R
        self.recon_model = recon_model
        self.mask_method = mask_method
        self.lr = lr
        self.center_region = center_region
        self.learn_R = learn_R

        self.sigmoid_slope_1 = sigmoid_slope1
        self.sigmoid_slope_2 = sigmoid_slope2
        self.lambda_param = lambda_param
        self.prob_method = prob_method
        self.fd_param = fd_param

        if prob_method == 'loupe':
            if warm_start: 
                O = gen_pdf_bern(image_size[1], image_size[2], 1/R, 8, center_region).astype(np.float32)
                O = torch.from_numpy(np.tile(O[np.newaxis, :, :], (image_size[0], 1, 1)))
                O = O/(O.max() + 1e-3)
            else:
                O = torch.rand(image_size)*(1 - 2e-2) + 1e-2
            self.sampling_weights = nn.Parameter(-torch.log((1/O) - 1) / self.sigmoid_slope_1)
        elif prob_method == 'line_loupe':
            O = torch.rand((image_size[0], image_size[2]))*(1 - 2e-2) + 1e-2 
            self.sampling_weights = nn.Parameter(-torch.log((1/O) - 1) / self.sigmoid_slope_k)
        elif prob_method == 'gubmel':
            self.sampling_weights = nn.Parameter(torch.rand(image_size))

        if self.learn_R: 
            self.R_value = nn.Parameter(torch.full((image_size[0],), float(self.R)))
        else: 
            self.R_value = torch.full((image_size[0],), float(self.R))


    def training_step(self, batch, batch_idx):
        k_space = batch['fs_k_space']
        sampling_mask, under_k, estimate_k = self.pass_through_model(k_space)
        #sampling_mask [batch, contrast, channel, height, width]

        loss = L1L2Loss(torch.view_as_real(estimate_k), torch.view_as_real(k_space)) 
        loss = loss + self.lambda_param * torch.sum(torch.any(1 - sampling_mask[:, :, 0, :, :], dim=1)) / k_space.shape[0]
        loss = loss + self.fd_param * torch.pow(self.sampling_weights.diff(dim=-1), 2).sum() / k_space.shape[0]
        loss = loss + self.fd_param * self.sampling_weights.diff(dim=-2).pow(2).sum() / k_space.shape[0]

        self.log("train/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.plot_images(k_space, 'train') 
        return loss



    def validation_step(self, batch, batch_idx):
        k_space = batch['fs_k_space']
        sampling_mask, under_k, estimate_k = self.pass_through_model(k_space)

        loss = L1L2Loss(torch.view_as_real(estimate_k), torch.view_as_real(k_space))

        estimated_img = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)
        img = root_sum_of_squares(ifft_2d_img(k_space), coil_dim=2)
        ssim_val = 0
        for i in range(img.shape[1]):
            ssim_val += ssim(img[:, [i], :, :], estimated_img[:, [i], :, :], device=self.device)

        self.log("val/val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/ssim", ssim_val/img.shape[1], on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.plot_images(k_space, 'val') 


    def forward(self, batch): 
        k_space = batch['fs_k_space']
        _, _, estimate_k = self.pass_through_model(k_space)
        return estimate_k

    def norm_prob(self, probability):
        if self.learn_R:
            cur_R = self.R_value * (self.R/self.R_value.mean())
        else:
            cur_R = self.R_value

        if self.prob_method == 'loupe' or self.prob_method == 'gumbel':
            center = [self.image_size[1]//2, self.image_size[2]//2]

            center_bb_x = slice(center[0]-self.center_region//2,center[0]+self.center_region//2)
            center_bb_y = slice(center[1]-self.center_region//2,center[1]+self.center_region//2)
            
            probability_sum = []

            # create acs mask of zeros for acs box and zeros elsewhere
            center_mask = torch.ones(self.image_size[1], self.image_size[2], device=self.device)
            center_mask[center_bb_y, center_bb_x] = 0
            for i in range(len(probability)):
                probability[i] = probability[i] * center_mask
                probability_sum.append(torch.sum(probability[i], dim=[-1, -2]))

            
            for i in range(len(probability_sum)):
                probability_total = self.image_size[-1] * self.image_size[-2]/ cur_R[i]
                probability_total = probability_total - self.center_region ** 2
                if probability_sum[i] > probability_total:
                    scaling_factor = probability_total / probability_sum[i]

                    assert scaling_factor <= 1 and scaling_factor >= 0
                    assert not torch.isnan(scaling_factor)

                    probability[i] = (probability[i] * scaling_factor)
                else:
                    print('Inverse prob scaling!')
                    inverse_total = self.image_size[1]*self.image_size[2]*(1 - 1/cur_R[i])
                    inverse_sum = (self.image_size[1]*self.image_size[2]) - probability_sum[i] - self.center_region**2
                    scaling_factor = inverse_total / inverse_sum

                    assert scaling_factor <= 1 and scaling_factor >= 0
                    assert not torch.isnan(scaling_factor)

                    inv_prob = (1 - probability[i])*scaling_factor
           
            # acs box is now ones and everything else is zeros
            for i in range(len(probability)):
                probability[i][center_bb_y, center_bb_x] = 1

        elif self.prob_method == 'line_loupe':
            center = self.image_size[2]//2
            center_bb_x = [center-self.center_region//2,center+self.center_region//2]

            probability_sum = []
            center_mask = torch.ones(self.image_size[2], device=self.device)
            center_mask[center_bb_x[0]:center_bb_x[1]] = 0
            for i in range(len(probability)):
                probability[i] = probability[i] * center_mask
                probability_sum.append(torch.sum(probability[i], dim=[-1]) - probability[i][center_bb_x[0]:center_bb_x[1]].sum())


            for i in range(len(probability_sum)):
                probability_total = self.image_size[2] / cur_R[i]
                probability_total -= self.center_region
                # if probability sum is greater than total scaling factor will 
                # be less than 1 so we can multiply
                if probability_sum[i] > probability_total: 
                    scaling_factor = probability_total/probability_sum[i]
                    probability[i] = probability[i] * scaling_factor
                else:
                # Scaling factor will be greater than 1 so we need to go into the 
                # inverse of probs
                    inverse_total = self.image_size[2]*(1 - 1/cur_R[i])
                    inverse_sum = self.image_size[2] - probability_sum[i] - self.center_region
                    scaling_factor = inverse_total / inverse_sum
                    inv_prob = (1 - probability[i])*scaling_factor
                    probability[i] = 1 - inv_prob

            center_box = 1-center_mask
            for i in range(len(probability)):
                probability[i] = probability[i] + center_box
        else:
            raise ValueError(f'No prob method found for {self.prob_method}')
        
        return probability



    def pass_through_model(self, k_space, deterministic=False):
        # Calculate probability and normalize

        assert not torch.isnan(self.sampling_weights).any() 
        if 'loupe' in self.prob_method:
            probability = [torch.sigmoid(sampling_weights * self.sigmoid_slope_1) for sampling_weights in self.sampling_weights]
            assert all((probs.min() >= 0 for probs in probability)), f'Probability should be greater than 1 but found {[prob.min() for prob in probability]}'
            assert all((probs.max() <= 1 for probs in probability)), f'Probability should be less than 1 but found {[prob.max() for prob in probability]}'
        elif self.prob_method == 'gumbel': 
            probability = self.sampling_weights
        else:
            raise TypeError('prob_method should be loupe or gumbel')

        norm_probability = self.norm_prob(probability)
        norm_probability = torch.stack(norm_probability, dim=0)
        
        # make sure nothing is nan 
        assert not torch.isnan(norm_probability).any()
        # make sure acs box is ones
        center_x, center_y = norm_probability.shape[1]//2, norm_probability.shape[2]//2
        acs_x = slice(center_x-self.center_region//2, center_x+self.center_region//2)
        acs_y = slice(center_y-self.center_region//2, center_y+self.center_region//2)
        acs_box = norm_probability[:, acs_y, acs_x]
        torch.testing.assert_close(acs_box, torch.ones(norm_probability.shape[0], self.center_region, self.center_region, device=self.device))
    
        # ensure acs line probabilities are really large so there is no change that they aren't sampled
        norm_probability[:, acs_y, acs_x] = norm_probability[:, acs_y, acs_x] * 10
        

        if self.prob_method == 'loupe':
            activation = norm_probability - torch.rand((k_space.shape[0],) + self.image_size, device=self.device)
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
    
        # Estimate k-space using the model
        estimate_k = self.recon_model({'input': under_k, 'mask': sampling_mask})
    
        return sampling_mask, under_k, estimate_k


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 6000, eta_min=1e-2) 
        return [optimizer], [scheduler]

    def plot_images(self, k_space, mode='train'):
        with torch.no_grad():
            sampling_mask, under_k, estimate_k = self.pass_through_model(k_space)

            super().plot_images(under_k, estimate_k, k_space, sampling_mask, mode)
            probability = torch.sigmoid(self.sampling_weights * self.sigmoid_slope_1) 

            tensorboard = self.logger.experiment

            sense_maps = self.recon_model.model.sens_model(under_k, sampling_mask.expand_as(k_space))
            masked_k = self.recon_model.model.sens_model.mask(under_k, sampling_mask.expand_as(k_space))
            masked_k = masked_k[0, 0, [0], :, :].abs()/(masked_k[0, 0, [0], :, :].abs().max()/20)
            sense_maps = sense_maps[0, 0, :, :, :].unsqueeze(1).abs()

            tensorboard.add_images(mode + '/sense_maps', sense_maps/sense_maps.max(), self.current_epoch)
            tensorboard.add_image(mode + '/sense_mask', masked_k, self.current_epoch)
                
            if probability.ndim == 2:
                probability = einops.repeat(probability, 'c chan h -> c chan h w', w=probability.shape[1])
            tensorboard.add_images(mode + '/probability', probability.unsqueeze(1), self.current_epoch)

            if isinstance(self.loggers, list): 

                wandb_logger = None
                for logger in self.loggers: 
                    if isinstance(logger, WandbLogger):
                        wandb_logger = logger
                if wandb_logger:
                    assert isinstance(wandb_logger, WandbLogger)
                    wandb_logger.log_image(mode + '/probability', np.split(probability.cpu().numpy(), probability.shape[0], 0))
                    wandb_logger.log_image(mode + '/sense_maps', np.split(sense_maps.cpu().numpy()/sense_maps.max().item(), sense_maps.shape[0], 0))
                    wandb_logger.log_image(mode + '/masked_k', [masked_k.clamp(0, 1).cpu().numpy()])
                    cur_R = self.R_value * (self.R/self.R_value.mean())
                    for i in range(len(self.contrast_order)):
                        contrast = self.contrast_order[i]
                        self.log(mode + "/R_Value_" + contrast, cur_R[i], on_step=False, on_epoch=True, logger=True)




    def sample_gumbel(self, size, eps=1e-20): 
        U = torch.rand(size, device=self.device)
        return -torch.log(-torch.log(U + eps) + eps)
