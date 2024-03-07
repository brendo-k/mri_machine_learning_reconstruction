import torch.nn as nn
import numpy as np
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
import einops

from ml_recon.losses import L1L2Loss
from ml_recon.pl_modules.pl_model import plReconModel

class LOUPE(plReconModel):
    def __init__(
            self, 
            recon_model,
            image_size, 
            learned_R, 
            contrast_order, 
            center_region = 10,
            prob_method='loupe', 
            mask_method='all',
            sigmoid_slope1 = 5,
            sigmoid_slope2 = 200,
            lr=1e-2,
            lambda_param = 0,
            ):
        super().__init__(contrast_order=contrast_order)
        self.save_hyperparameters(ignore='recon_model')
        self.image_size = image_size
        self.contrast_order = contrast_order
        self.R = learned_R
        self.recon_model = recon_model
        self.mask_method = mask_method
        self.lr = lr
        self.center_region = center_region

        self.sigmoid_slope_1 = sigmoid_slope1
        self.sigmoid_slope_2 = sigmoid_slope2
        self.lambda_param = lambda_param
        self.prob_method = prob_method

        if prob_method == 'loupe':
            O = [torch.rand((image_size[1], image_size[2]))*(1 - 2e-2) + 1e-2 for _ in range(len(self.contrast_order))]
            self.sampling_weights = nn.ParameterList([nn.Parameter(-torch.log(1/samp - 1) / self.sigmoid_slope_1) for samp in O])
        elif prob_method == 'line_loupe':
            O = [torch.rand(image_size[2])*(1 - 2e-2) + 1e-2 for _ in range(len(self.contrast_order))]
            self.sampling_weights = nn.ParameterList([nn.Parameter(-torch.log(1/samp - 1) / self.sigmoid_slope_1) for samp in O])
        elif prob_method == 'gubmel':
            self.sampling_weights = nn.Parameter(torch.rand(image_size))


    def training_step(self, batch, batch_idx):
        under, k_space = batch
        sampling_mask, under_k, estimate_k = self.pass_through_model(k_space)
        #sampling_mask [batch, contrast, channel, height, width]

        loss = L1L2Loss(torch.view_as_real(estimate_k), torch.view_as_real(k_space)) + self.lambda_param * torch.sum(1 - sampling_mask[:, :, 0, :, :])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.plot_images(k_space, 'train') 
        return loss



    def validation_step(self, batch, batch_idx):
        under, k_space = batch
        sampling_mask, under_k, estimate_k = self.pass_through_model(k_space)

        loss = L1L2Loss(torch.view_as_real(estimate_k), torch.view_as_real(k_space))

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.plot_images(k_space, 'val') 


    def forward(self, batch, _): 
        _, k_space = batch
        _, _, estimate_k = self.pass_through_model(k_space)
        return estimate_k

    def norm_prob(self, probability):
        if self.prob_method == 'loupe' or self.prob_method == 'gumbel':
            center = [self.image_size[1]//2, self.image_size[2]//2]

            center_bb_x = [center[0]-self.center_region//2,center[0]+self.center_region//2]
            center_bb_y = [center[1]-self.center_region//2,center[1]+self.center_region//2]
            
            probability_sum = []
            center_mask = torch.ones(self.image_size[1], self.image_size[2], device=self.device)
            center_mask[center_bb_x[0]:center_bb_x[1], center_bb_y[0]:center_bb_y[1]] = 0
            for i in range(len(probability)):
                probability[i] = probability[i] * center_mask
                probability_sum.append(torch.sum(probability[i], dim=[-1, -2]))

            probability_total = self.image_size[-1] * self.image_size[-2]/ self.R
            probability_total = probability_total - self.center_region ** 2
            
            for i in range(len(probability_sum)):
                if probability_sum[i] > probability_total:
                    scaling_factor = probability_total / probability_sum[i]

                    assert scaling_factor <= 1 and scaling_factor >= 0
                    assert not torch.isnan(scaling_factor)

                    probability[i] = (probability[i] * scaling_factor)
                else:
                    print('Inverse prob scaling!')
                    inverse_total = torch.prod(probability.shape[-2:])*(1 - 1/self.R)
                    inverse_sum = torch.prod(probability.shape[-2:]) - probability_sum[i] - self.center_region**2
                    scaling_factor = inverse_total / inverse_sum

                    assert scaling_factor <= 1 and scaling_factor >= 0
                    assert not torch.isnan(scaling_factor)

                    inv_prob = (1 - probability[i])*scaling_factor
                    probability[i] = (1 - inv_prob)
            
            center_box = 1 - center_mask
            for i in range(len(probability)):
                probability[i] = probability[i] + center_box 

        elif self.prob_method == 'line_loupe':
            center = self.image_size[2]//2
            center_bb_x = [center-self.center_region//2,center+self.center_region//2]

            probability_sum = []
            center_mask = torch.ones(self.image_size[2], device=self.device)
            center_mask[center_bb_x[0]:center_bb_x[1]] = 0
            for i in range(len(probability)):
                probability[i] = probability[i] * center_mask
                probability_sum.append(torch.sum(probability[i], dim=[-1]) - probability[i][center_bb_x[0]:center_bb_x[1]].sum())

            probability_total = self.image_size[2] / self.R
            probability_total -= self.center_region

            for i in range(len(probability_sum)):
                # if probability sum is greater than total scaling factor will 
                # be less than 1 so we can multiply
                if probability_sum[i] > probability_total: 
                    scaling_factor = probability_total/probability_sum[i]
                    probability[i] = probability[i] * scaling_factor
                else:
                # Scaling factor will be greater than 1 so we need to go into the 
                # inverse of probs
                    inverse_total = self.image_size[2]*(1 - 1/self.R)
                    inverse_sum = self.image_size[2] - probability_sum - self.center_region
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

        assert not any((torch.isnan(samp_weights).any() for samp_weights in self.sampling_weights))
        if 'loupe' in self.prob_method:
            probability = [torch.sigmoid(samp_weights * self.sigmoid_slope_1) for samp_weights in self.sampling_weights]
            assert all((probs.min() >= 0 for probs in probability)), f'Probability should be greater than 1 but found {[prob.min() for prob in probability]}'
            assert all((probs.max() <= 1 for probs in probability)), f'Probability should be less than 1 but found {[prob.max() for prob in probability]}'
        elif self.prob_method == 'gumbel': 
            probability = self.sampling_weights
        else:
            raise TypeError('prob_method should be loupe or gumbel')

        prob_list = self.norm_prob(probability)
        norm_probability = torch.stack(prob_list, dim=0)

        assert not torch.isnan(norm_probability).any()

        if self.prob_method == 'loupe':
            activation = norm_probability - torch.rand((k_space.shape[0],) + self.image_size, device=self.device)
            # slope increase as epochs continue
            #if self.current_epoch > 50:
            #    self.sigmoid_slope_2 += 2
            #if self.current_epoch > 50: 
            #    self.sigmoid_slope_1 *= 1.1
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
        estimate_k = self.recon_model((under_k, k_space), sampling_mask)
    
        return sampling_mask, under_k, estimate_k


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 6000, eta_min=1e-2) 
        return [optimizer], [scheduler]

    def plot_images(self, k_space, mode='train'):
        with torch.no_grad():
            sampling_mask, under_k, estimate_k = self.pass_through_model(k_space)
            super().plot_images((under_k, k_space), sampling_mask, mode)

            tensorboard = self.logger.experiment

            sense_maps = self.recon_model.model.sens_model(under_k, sampling_mask.expand_as(k_space))
            masked_k = self.recon_model.model.sens_model.mask(under_k, sampling_mask.expand_as(k_space))
            masked_k = masked_k[0, 0, [0], :, :].abs()/(masked_k[0, 0, [0], :, :].abs().max()/20)
            sense_maps = sense_maps[0, 0, :, :, :].unsqueeze(1).abs()

            tensorboard.add_images(mode + '/sense_maps', sense_maps/sense_maps.max(), self.current_epoch)
            tensorboard.add_image(mode + '/sense_mask', masked_k, self.current_epoch)
                
            weight_list = [weight for weight in self.sampling_weights]
            sampling_weights = torch.stack(weight_list, dim=0)
            weights = sampling_weights.unsqueeze(1)
            if sampling_weights.ndim == 2:
                weights = einops.repeat(weights, 'c chan h -> c chan h w', w=sampling_weights.shape[1])
            tensorboard.add_images(mode + '/sample_weights', weights/weights.max(), self.current_epoch)

            if isinstance(self.loggers, list): 

                wandb_logger = None
                for logger in self.loggers: 
                    if isinstance(logger, WandbLogger):
                        wandb_logger = logger
                if wandb_logger:
                    assert isinstance(wandb_logger, WandbLogger)
                    wandb_logger.log_image(mode + '/sample_weights', np.split(weights.cpu().numpy(), weights.shape[0], 0))



    def sample_gumbel(self, size, eps=1e-20): 
        U = torch.rand(size, device=self.device)
        return -torch.log(-torch.log(U + eps) + eps)
