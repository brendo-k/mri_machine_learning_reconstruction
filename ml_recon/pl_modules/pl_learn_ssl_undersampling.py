import torch.nn as nn
import numpy as np
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
import einops
from typing import List

from ml_recon.losses import L1L2Loss
from ml_recon.dataset.undersample import scale_pdf, gen_pdf_bern, gen_pdf_columns
from ml_recon.pl_modules.pl_model import plReconModel
from ml_recon.utils.evaluate import nmse, ssim, psnr
from ml_recon.utils import ifft_2d_img, root_sum_of_squares
from ml_recon.pl_modules.pl_varnet import pl_VarNet

class LearnedSSLLightning(plReconModel):
    def __init__(
            self, 
            image_size, 
            learned_R: float, 
            contrast_order: List[str], 
            center_region:int = 10,
            prob_method:str = 'loupe', 
            sigmoid_slope1:float = 5.0,
            sigmoid_slope2:float = 200.0,
            lr:float = 1e-2,
            warm_start:bool = False, 
            learn_R:bool = False,
            ssim_scaling = 1e-4,
            normalize_k_space_energy: float = 0.0,
            lambda_scaling: float = 0.0, 
            pass_all_data: bool = False,
            pass_inverse_data: bool = False,
            ):
        super().__init__(contrast_order=contrast_order)
        self.save_hyperparameters(ignore='recon_model')

        self.recon_model = pl_VarNet(contrast_order=contrast_order)
        self.image_size = image_size
        self.contrast_order = contrast_order
        self.R = learned_R
        self.lr = lr
        self.center_region = center_region
        self.learn_R = learn_R
        self.sigmoid_slope_1 = sigmoid_slope1
        self.sigmoid_slope_2 = sigmoid_slope2
        self.prob_method = prob_method
        self.ssim_scaling = ssim_scaling
        self.lambda_scaling = lambda_scaling
        self.norm_k_space = normalize_k_space_energy
        self.pass_all_data = pass_all_data
        self.pass_inverse_data = pass_inverse_data

        self.R_value = torch.full((image_size[0],), float(self.R))
        self.R_freeze = [False for _ in range(len(contrast_order))]

        if self.learn_R: 
            self.R_value = nn.Parameter(torch.full((image_size[0],), float(self.R)))


        if prob_method == 'loupe':
            if warm_start: 
                init_prob = gen_pdf_bern(image_size[1], image_size[2], 1/self.R, 8, center_region).astype(np.float32)
                init_prob = torch.from_numpy(np.tile(init_prob[np.newaxis, :, :], (image_size[0], 1, 1)))
                init_prob = init_prob/(init_prob.max() + 2e-4) + 1e-4
            else:
                init_prob = torch.zeros(image_size) + 0.5
            self.sampling_weights = nn.Parameter(-torch.log((1/init_prob) - 1) / self.sigmoid_slope_1)

        elif prob_method == 'line_loupe':
            O = torch.rand((image_size[0], image_size[2]))*(1 - 2e-2) + 1e-2 
            self.sampling_weights = nn.Parameter(-torch.log((1/O) - 1) / self.sigmoid_slope_k)

    def training_step(self, batch, batch_idx):
        undersampled = batch['input']
        initial_mask = undersampled != 0

        nbatch, contrast, coil, h, w = undersampled.shape
        
        first_sampling_mask = self.get_mask(self.sampling_weights, nbatch, mask_center=True)
        first_sampling_mask = first_sampling_mask.unsqueeze(2)
        mask_lambda = initial_mask * first_sampling_mask
        inverse_mask = 1 - first_sampling_mask.clone()
        mask_inverse = initial_mask * inverse_mask

        estimate_lambda = self.recon_model({'input': undersampled*mask_lambda, 'mask': mask_lambda})

        loss_lambda = L1L2Loss(torch.view_as_real(undersampled*mask_inverse), torch.view_as_real(estimate_lambda*mask_inverse)) 
        estimate_lambda = estimate_lambda * inverse_mask + undersampled * mask_lambda
        image1 = root_sum_of_squares(ifft_2d_img(estimate_lambda), coil_dim=2) 
        loss = loss_lambda 
        loss += self.norm_k_space * (undersampled.abs().max() - mask_lambda * undersampled.abs()).mean()
        
        inverse_image = None
        if self.pass_inverse_data:
            mask_inverse_w_acs = mask_inverse.clone()
            mask_inverse_w_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 1
            estimate_inverse = self.recon_model({'input': undersampled*mask_inverse_w_acs, 'mask': mask_inverse_w_acs})
            mask_lambda_wo_acs = mask_lambda.clone()
            mask_lambda_wo_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 0
            estimate_inverse = estimate_inverse * mask_lambda_wo_acs + undersampled * mask_inverse_w_acs
            inverse_image = root_sum_of_squares(ifft_2d_img(estimate_inverse), coil_dim=2) 
            
            # calculate loss
            loss_inverse = self.lambda_scaling * L1L2Loss(torch.view_as_real(undersampled*mask_lambda_wo_acs), torch.view_as_real(estimate_inverse*mask_lambda_wo_acs)) 
            b, c, h, w = image1.shape
            image1 = image1.reshape(-1, image1.shape[-2], image1.shape[-1]).unsqueeze(1)
            inverse_image = inverse_image.reshape(-1, image1.shape[-2], image1.shape[-1]).unsqueeze(1)
            ssim_loss = 1 - ssim(image1, inverse_image, self.device)
            image1 = image1.reshape(b, c, h, w)
            inverse_image = inverse_image.reshape(b, c, h, w)

            ssim_loss *= self.ssim_scaling
            self.log("train/loss_inverse", loss_inverse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train/ssim_loss", ssim_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            if batch_idx == 0:
                self.logger.log_image('train/estimate_inverse', np.split(inverse_image[0].abs()/inverse_image[0].abs().max(),image1.shape[1], 0))
            loss += loss_inverse
            loss += ssim_loss


        if self.pass_all_data:
            estimate_full = self.recon_model({'input': undersampled, 'mask': initial_mask})
            estimate_full = estimate_full * ~initial_mask + undersampled
            image_full = root_sum_of_squares(ifft_2d_img(estimate_full), coil_dim=2) 

            b, c, h, w = image1.shape
            image1 = image1.reshape(-1, image1.shape[-2], image1.shape[-1]).unsqueeze(1)
            image_full = image_full.reshape(-1, image1.shape[-2], image1.shape[-1]).unsqueeze(1)
            ssim_loss_full = 1 - ssim(image1, image_full, self.device)
            
            if self.pass_inverse_data:
                assert inverse_image is not None, "should exist!"
                inverse_image = inverse_image.reshape(-1, image1.shape[-2], image1.shape[-1]).unsqueeze(1)
                ssim_loss_full_inverse = 1 - ssim(inverse_image, image_full, self.device)
                inverse_image.reshape(b, c, h, w)
                loss += ssim_loss_full_inverse * self.ssim_scaling

            image1 = image1.reshape(b, c, h, w)
            image_full = image_full.reshape(b, c, h, w)
            loss += ssim_loss_full * self.ssim_scaling



        self.log("train/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss_lambda", loss_lambda, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            with torch.no_grad():
                image1 = image1.detach().cpu()
                wandb_logger = self.logger
                initial_mask = initial_mask[0, :, 0, :, :]
                wandb_logger.log_image('train' + '/initial_mask', np.split(initial_mask.cpu().detach().numpy(), initial_mask.shape[0], 0))
                mask_lambda = mask_lambda[0, :, 0, : ,:]
                mask_inverse = mask_inverse[0, :, 0, : ,:]
                first_sampling_mask = first_sampling_mask[0, :, :, :]
                wandb_logger.log_image('train/omega_lambda', np.split(mask_lambda.cpu().detach().numpy(), mask_lambda.shape[0], 0))
                wandb_logger.log_image('train/omega_(1-lambda)', np.split(mask_inverse.cpu().detach().numpy(), mask_inverse.shape[0], 0))
                wandb_logger.log_image('train/estimate_lambda', np.split(image1[0].abs()/image1[0].abs().max(),image1.shape[1], 0))
                wandb_logger.log_image('train/first_mask', np.split(first_sampling_mask.cpu().detach().numpy(), mask_lambda.shape[0], 0))
                wandb_logger.log_image('train/initial_mask', np.split(initial_mask.cpu().detach().numpy(), mask_lambda.shape[0], 0))

                probability = [torch.sigmoid(sampling_weights * self.sigmoid_slope_1) for sampling_weights in self.sampling_weights]
                R_value = self.norm_R(self.R_value)
                probability = self.norm_prob(probability, R_value, mask_center=True)
                probability = torch.stack(probability, dim=0)
                wandb_logger.log_image('train/probability', np.split(probability.abs(), image1.shape[1], 0))
        return loss



    def validation_step(self, batch, batch_idx):
        under = batch['input']
        fs_k_space = batch['fs_k_space']
        initial_mask = under != 0 

        nbatch, contrast, coil, h, w = under.shape
        
        first_sampling_mask = self.get_mask(self.sampling_weights, nbatch, mask_center=True)
        first_sampling_mask = first_sampling_mask.unsqueeze(2)
        inverse_mask = 1 - first_sampling_mask

        mask_lambda = initial_mask * first_sampling_mask
        mask_inverse = initial_mask * inverse_mask
        mask_inverse_w_acs = mask_inverse.clone()
        mask_inverse_w_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 1
        mask_lambda_wo_acs = mask_lambda.clone()
        mask_lambda_wo_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 1

        estimate_lambda = self.recon_model({'input': under*mask_lambda, 'mask': mask_lambda})
        estimate_inverse = self.recon_model({'input': under*mask_inverse_w_acs, 'mask': mask_inverse_w_acs}) 
        estimate_full = self.recon_model({'input': under, 'mask': initial_mask})

        loss_inverse = L1L2Loss(torch.view_as_real(estimate_inverse*mask_lambda_wo_acs), torch.view_as_real(under * mask_lambda_wo_acs)) 
        loss_lambda = L1L2Loss(torch.view_as_real(estimate_lambda*mask_inverse), torch.view_as_real(under * mask_inverse)) 

        estimate_lambda = estimate_lambda * ~initial_mask + under
        estimate_inverse = estimate_inverse * ~initial_mask + under
        estimate_full = estimate_full * ~initial_mask + under

        est_lambda_img = root_sum_of_squares(ifft_2d_img(estimate_lambda), coil_dim=2)
        est_inverse_img = root_sum_of_squares(ifft_2d_img(estimate_inverse), coil_dim=2)
        estimated_img = root_sum_of_squares(ifft_2d_img(estimate_full), coil_dim=2)
        fully_sampled_img = root_sum_of_squares(ifft_2d_img(fs_k_space), coil_dim=2)

        wandb_logger = self.logger

        ssim_loss = 0
        for i in range(fully_sampled_img.shape[1]):
            ssim_loss += ssim(est_lambda_img[:, [i], :, :], est_inverse_img[:, [i], :, :], device=self.device)

        ssim_val = 0
        for i in range(fully_sampled_img.shape[1]):
            ssim_val += ssim(fully_sampled_img[:, [i], :, :], estimated_img[:, [i], :, :], device=self.device)

        self.log("val/val_loss_inverse", loss_inverse, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/val_loss_lambda", loss_lambda, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/total_ssim", ssim_val/fully_sampled_img.shape[1], on_epoch=True, prog_bar=True, logger=True)
        self.log("val/ssim_loss", self.ssim_scaling * (ssim_loss/fully_sampled_img.shape[1]), on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            wandb_logger.log_image('val/estimate_lambda', np.split(est_lambda_img[0].cpu().numpy()/est_lambda_img[0].max().cpu().numpy(), est_lambda_img.shape[1], 0))
            wandb_logger.log_image('val/estimate_inverse', np.split(est_inverse_img[0].cpu().numpy()/est_inverse_img[0].max().cpu().numpy(), est_inverse_img.shape[1], 0))
            wandb_logger.log_image('val/estimate_full', np.split(estimated_img[0].cpu().numpy()/estimated_img[0].max().cpu().numpy(), est_inverse_img.shape[1], 0))
            wandb_logger.log_image('val/ground_truth', np.split(fully_sampled_img[0].cpu().numpy()/fully_sampled_img[0].max().cpu().numpy(), est_inverse_img.shape[1], 0))


    def test_step(self, batch, batch_idx):
        undersampled = batch['input']
        k_space = batch['fs_k_space']

        first_sampling_mask = undersampled != 0


        estimate_k = self.recon_model({'input': undersampled, 'mask': first_sampling_mask})
        estimate_k = estimate_k + ~first_sampling_mask + undersampled

        super().test_step((estimate_k, k_space), None)

    

    def forward(self, batch): 
        k_space = batch['fs_k_space']
        _, _, _, estimate_k = self.pass_through_model(k_space)
        return estimate_k

    # takes an array of R values and normalizes it to the desired R value
    def norm_R(self, R):
        if self.learn_R:
            inverse = [1/R_val for R_val, freeze in zip(R, self.R_freeze) if not freeze]
            cur_R = []
            for R_val, freeze in zip(R, self.R_freeze):
                # normalize R_value unless it is forzen
                if not freeze:
                    R_val = (R_val * sum(inverse) / (len(inverse)/self.R))
                cur_R.append(R_val)
        else: 
            cur_R = R
                
        return cur_R


    def norm_prob(self, probability:List[torch.Tensor], cur_R, center_region=10, mask_center=False):
        image_shape = probability[0].shape
        if self.prob_method == 'loupe':
            center = [image_shape[0]//2, image_shape[1]//2]

            center_bb_x = slice(center[0]-center_region//2,center[0]+center_region//2)
            center_bb_y = slice(center[1]-center_region//2,center[1]+center_region//2)
            
            probability_sum = []

            # create acs mask of zeros for acs box and zeros elsewhere
            center_mask = torch.ones(image_shape, device=probability[0].device)
            if mask_center:
                center_mask[center_bb_y, center_bb_x] = 0

            for i in range(len(probability)):
                probability[i] = probability[i] * center_mask
                probability_sum.append(torch.sum(probability[i], dim=[-1, -2]))

            
            for i in range(len(probability_sum)):
                probability_total = image_shape[-1] * image_shape[-2]/ cur_R[i]
                if mask_center:
                    probability_total -= center_region ** 2

                if probability_sum[i] > probability_total:
                    scaling_factor = probability_total / probability_sum[i]

                    assert scaling_factor <= 1 and scaling_factor >= 0
                    assert not torch.isnan(scaling_factor)

                    probability[i] = (probability[i] * scaling_factor)
                else:
                    inverse_total = image_shape[1]*image_shape[0]*(1 - 1/cur_R[i])
                    inverse_sum = (image_shape[1]*image_shape[0]) - probability_sum[i] 
                    if mask_center:
                        inverse_sum -= center_region**2
                    scaling_factor = inverse_total / inverse_sum

                    assert scaling_factor <= 1 and scaling_factor >= 0
                    assert not torch.isnan(scaling_factor)

                    inv_prob = (1 - probability[i])*scaling_factor
                    probability[i] = 1 - inv_prob
           
            # acs box is now ones and everything else is zeros
            if mask_center:
                for i in range(len(probability)):
                    probability[i][center_bb_y, center_bb_x] = 1

        elif self.prob_method == 'line_loupe':
            center = image_shape[1]//2
            center_bb_x = [center-self.center_region//2,center+self.center_region//2]

            probability_sum = []
            center_mask = torch.ones(self.image_size[2], device=self.device)
            if mask_center:
                center_mask[center_bb_x[0]:center_bb_x[1]] = 0

            for i in range(len(probability)):
                probability[i] = probability[i] * center_mask
                probability_sum.append(torch.sum(probability[i], dim=[-1]) - probability[i][center_bb_x[0]:center_bb_x[1]].sum())


            for i in range(len(probability_sum)):
                probability_total = image_shape[1] / cur_R[i]
                if mask_center:
                    probability_total -= center_region

                # if probability sum is greater than total scaling factor will 
                # be less than 1 so we can multiply
                if probability_sum[i] > probability_total: 
                    scaling_factor = probability_total/probability_sum[i]
                    probability[i] = probability[i] * scaling_factor
                else:
                # Scaling factor will be greater than 1 so we need to go into the 
                # inverse of probs
                    inverse_total = image_shape[1]*(1 - 1/cur_R[i])
                    inverse_sum = image_shape[1] - probability_sum[i] 
                    if mask_center:
                        inverse_sum -= center_region
                    scaling_factor = inverse_total / inverse_sum
                    inv_prob = (1 - probability[i])*scaling_factor
                    probability[i] = 1 - inv_prob

            center_box = 1-center_mask
            for i in range(len(probability)):
                probability[i] = probability[i] + center_box
        else:
            raise ValueError(f'No prob method found for {self.prob_method}')
        
        assert all((torch.isclose(probs.mean(), 1/R, atol=0.01, rtol=0) for probs, R in zip(probability, cur_R))), f'Probability should be equal to R {[prob.mean() for prob in probability]}'
        return probability


    def get_mask(self, sampling_weights, batch_size, mask_center=False, deterministic=False):
        # Calculate probability and normalize

        assert not torch.isnan(sampling_weights).any(), "sampling weights shouldn't be nan!"
        assert sampling_weights.shape == self.image_size, "sampling weights should match the image size" 

        if 'loupe' in self.prob_method:
            probability = [torch.sigmoid(sampling_weights * self.sigmoid_slope_1) for sampling_weights in sampling_weights]
            assert all((probs.min() >= 0 for probs in probability)), f'Probability should be greater than 1 but found {[prob.min() for prob in probability]}'
            assert all((probs.max() <= 1 for probs in probability)), f'Probability should be less than 1 but found {[prob.max() for prob in probability]}'
        else:
            raise TypeError('prob_method should be loupe or gumbel')
        
        R_value = self.norm_R(self.R_value)
        norm_probability = self.norm_prob(probability, R_value, mask_center=mask_center)
        norm_probability = torch.stack(norm_probability, dim=0)
        
        # make sure nothing is nan 
        assert not torch.isnan(norm_probability).any()

        if mask_center:
            # make sure acs box is ones
            center_x, center_y = norm_probability.shape[1]//2, norm_probability.shape[2]//2
            acs_x = slice(center_x-self.center_region//2, center_x+self.center_region//2)
            acs_y = slice(center_y-self.center_region//2, center_y+self.center_region//2)
            acs_box = norm_probability[:, acs_y, acs_x]
            torch.testing.assert_close(acs_box, torch.ones(norm_probability.shape[0], self.center_region, self.center_region, device=self.device))
    
            # ensure acs line probabilities are really large so there is no change that they aren't sampled
            norm_probability[:, acs_y, acs_x] = norm_probability[:, acs_y, acs_x] * 10

        if self.prob_method == 'loupe':
            activation = norm_probability - torch.rand((batch_size,) + self.image_size, device=self.device)
        elif self.prob_method == 'line_loupe':
            activation = norm_probability - torch.rand((batch_size, self.image_size[0], self.image_size[2]), device=self.device)
            activation = einops.repeat(activation, 'b c w -> b c h w', h=self.image_size[-1])
        else:
            raise ValueError('No sampling method!')

        if deterministic:
            sampling_mask = (activation > 0).to(torch.float)
        else:
            sampling_mask = torch.sigmoid(activation * self.sigmoid_slope_2)

        # check to make sure sampling the correct R

        inverse = [1/R_val for R_val, freeze in zip(self.R_value, self.R_freeze) if not freeze]
        cur_R = []
        for R_val, freeze in zip(self.R_value, self.R_freeze):
            if freeze:
                cur_R.append(R_val)
            else:
                cur_R.append(R_val * sum(inverse) / (len(inverse)/self.R))

        for i in range(sampling_mask.shape[0]):
            for j in range(sampling_mask.shape[1]):
                assert (torch.isclose(sampling_mask[i, j].mean(), 1/cur_R[j], atol=0.10, rtol=0.00)), f'Should be close! Got {sampling_mask[i, j].mean()} and {1/cur_R[j]}'

        assert not torch.isnan(sampling_mask).any()
        # Ensure sampling mask values are within [0, 1]
        assert sampling_mask.min() >= 0 and sampling_mask.max() <= 1
    
        return sampling_mask
        

    def pass_through_model(self, k_space, deterministic=False):
        # Calculate probability and normalize
        
        first_sampling_mask = self.get_mask(self.sampling_weights, k_space.shape[0], deterministic=False, mask_center=True)
        # Apply the sampling mask to k_space
        
        num_coils = k_space.shape[2]
        first_sampling_mask = torch.tile(first_sampling_mask.unsqueeze(2), (1, 1, num_coils, 1, 1))
        assert first_sampling_mask.shape == k_space.shape, 'Sampling mask and k_space should have the same shape!'

        under_k = k_space * first_sampling_mask

        loss_mask = torch.ones_like(first_sampling_mask)
        ssl_sampling_mask = torch.ones_like(first_sampling_mask)
        input = under_k
        target = k_space
    
        # Estimate k-space using the model
        estimate_k = self.recon_model({'input': input, 'mask': first_sampling_mask * ssl_sampling_mask})
    
        return first_sampling_mask, loss_mask, target, estimate_k


    def plot_images(self, batch, mode='train'):
        k_space = batch['fs_k_space']
        with torch.no_grad():
            sampling_mask, loss_mask, target, estimate_k = self.pass_through_model(k_space)

            super().plot_images(k_space*sampling_mask, estimate_k, target, k_space, sampling_mask, mode)
            probability = [torch.sigmoid(sampling_weights * self.sigmoid_slope_1) for sampling_weights in self.sampling_weights]
            R_value = self.norm_R(self.R_value)
            probability = self.norm_prob(probability, R_value, mask_center=True)
            probability = torch.stack(probability, dim=0)

            sense_maps = self.recon_model.model.sens_model(k_space*sampling_mask, sampling_mask.expand_as(k_space))
            masked_k = self.recon_model.model.sens_model.mask(k_space*sampling_mask, sampling_mask.expand_as(k_space))
            masked_k = masked_k[0, 0, [0], :, :].abs()/(masked_k[0, 0, [0], :, :].abs().max()/20)
            sense_maps = sense_maps[0, 0, :, :, :].unsqueeze(1).abs()

            if probability.ndim == 2:
                probability = einops.repeat(probability, 'c chan h -> c chan h w', w=probability.shape[0])
            
            wandb_logger = self.logger
            wandb_logger.log_image(mode + '/probability', np.split(probability.cpu().numpy(), probability.shape[0], 0))
            inverse = [1/R_val for R_val, freeze in zip(self.R_value, self.R_freeze) if not freeze]
            cur_R = []
            for R_val, freeze in zip(self.R_value, self.R_freeze):
                if freeze:
                    cur_R.append(R_val)
                else:
                    cur_R.append(R_val * sum(inverse) / (len(inverse)/self.R))
            for i in range(len(self.contrast_order)):
                contrast = self.contrast_order[i]
                self.log(mode + "/R_Value_" + contrast, cur_R[i], on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 6000, eta_min=1e-2) 
        return [optimizer], [scheduler]
