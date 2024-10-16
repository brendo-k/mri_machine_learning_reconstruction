import torch.nn as nn
import numpy as np
import torch
import einops
from typing import List

from torchmetrics.functional.image import structural_similarity_index_measure as ssim 
from ml_recon.losses import L1L2Loss
from ml_recon.utils.undersample_tools import gen_pdf_bern
from ml_recon.pl_modules.pl_ReconModel import plReconModel
from ml_recon.utils.evaluate import nmse
from ml_recon.utils import ifft_2d_img, root_sum_of_squares
from ml_recon.models import VarNet_mc
from ml_recon.utils.kmax_relaxation import KMaxSoftmaxFunction
from ml_recon.utils.evaluate_over_contrasts import evaluate_over_contrasts

class LearnedSSLLightning(plReconModel):
    def __init__(
            self, 
            image_size, 
            learned_R: float, 
            contrast_order: List[str], 
            channels: int = 32,
            center_region:int = 10,
            prob_method:str = 'loupe', 
            sigmoid_slope1:float = 5.0,
            sigmoid_slope2:float = 200,
            lr:float = 1e-2,
            warm_start:bool = False, 
            learn_R:bool = False,
            ssim_scaling_set = 1e-4,
            ssim_scaling_full = 1e-4,
            ssim_scaling_inverse = 1e-4,
            normalize_k_space_energy: float = 0.0,
            lambda_scaling: float = 0.0, 
            pass_all_data: bool = False,
            pass_inverse_data: bool = False,
            supervised: bool = False,
            learn_sampling: bool = True
            ):
        super().__init__(contrast_order=contrast_order)
        self.save_hyperparameters(ignore='recon_model')

        self.recon_model = VarNet_mc(contrasts=len(contrast_order), chans=channels)
        self.image_size = image_size
        self.contrast_order = contrast_order
        self.R = learned_R
        self.lr = lr
        self.center_region = center_region
        self.learn_R = learn_R
        self.sigmoid_slope_1 = sigmoid_slope1
        self.sigmoid_slope_2 = sigmoid_slope2
        self.prob_method = prob_method
        self.ssim_scaling_set = ssim_scaling_set
        self.ssim_scaling_full = ssim_scaling_full
        self.ssim_scaling_inverse_full = ssim_scaling_inverse
        self.lambda_scaling = lambda_scaling
        self.norm_k_space = normalize_k_space_energy
        self.pass_all_data = pass_all_data
        self.supervised = supervised
        self.pass_inverse_data = pass_inverse_data

        self.R_value = torch.full((image_size[0],), float(self.R))
        self.R_freeze = [False for _ in range(len(contrast_order))]

        self.loss_func = L1L2Loss

        if self.learn_R: 
            self.R_value = nn.Parameter(torch.full((image_size[0],), float(self.R)))
        else: 
            self.R_value = torch.full((image_size[0],), float(self.R))

        if prob_method == 'loupe':
            if warm_start: 
                init_prob = gen_pdf_bern(image_size[1], image_size[2], 1/self.R, 8, center_region).astype(np.float32)
                init_prob = torch.from_numpy(np.tile(init_prob[np.newaxis, :, :], (image_size[0], 1, 1)))
                init_prob = init_prob/(init_prob.max() + 2e-4) + 1e-4
            else:
                init_prob = torch.zeros(image_size) + 0.5
            self.sampling_weights = nn.Parameter(-torch.log((1/init_prob) - 1) / self.sigmoid_slope_1, requires_grad=learn_sampling)

        elif prob_method == 'line_loupe':
            O = torch.rand((image_size[0], image_size[2]))*(1 - 2e-2) + 1e-2 
            self.sampling_weights = nn.Parameter(-torch.log((1/O) - 1) / self.sigmoid_slope_k, requires_grad=learn_sampling)

    def training_step(self, batch, batch_idx):
        if self.supervised: 
            return self.train_supervised_step(batch)

        undersampled = batch['input']
        initial_mask = (undersampled != 0).to(torch.float32)
        nbatch, contrast, coil, h, w = undersampled.shape

        lambda_set, inverse_set = self.split_into_lambda_loss_sets(initial_mask, nbatch)
        estimate_lambda = self.pass_through_model(undersampled, lambda_set)

        loss_lambda = self.loss_func(
                torch.view_as_real(undersampled*inverse_set), 
                torch.view_as_real(estimate_lambda*inverse_set)
                               ) 

        zero_filled = root_sum_of_squares(ifft_2d_img(undersampled), coil_dim=2) 
        scaling_factor = zero_filled.amax((-1, -2), keepdim=True)
        lambda_image = root_sum_of_squares(ifft_2d_img(estimate_lambda), coil_dim=2)/scaling_factor
        
        loss = loss_lambda 
        
        inverse_image = None
        if self.pass_inverse_data:
            # create new masks with inverted acs lines
            mask_inverse_w_acs = inverse_set.clone()
            mask_lambda_wo_acs = lambda_set.clone()
            mask_inverse_w_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 1
            mask_lambda_wo_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 0

            estimate_inverse = self.pass_through_model(undersampled, mask_inverse_w_acs)
            inverse_image = root_sum_of_squares(ifft_2d_img(estimate_inverse), coil_dim=2)/scaling_factor
            
            # calculate loss

            loss_inverse = self.loss_func(
                    torch.view_as_real(undersampled*mask_lambda_wo_acs), 
                    torch.view_as_real(estimate_inverse*mask_lambda_wo_acs)
                    )
            loss_inverse *= self.lambda_scaling # scale inverse loss by lambda_scaling

            b, c, h, w = lambda_image.shape
            lambda_image = lambda_image.reshape(b * c, 1, h, w)
            inverse_image = inverse_image.reshape(b * c, 1, h, w)
            ssim_loss = torch.tensor(1, device=self.device) - ssim(lambda_image, inverse_image, data_range=(1, 0))

            lambda_image = lambda_image.reshape(b, c, h, w)
            inverse_image = inverse_image.reshape(b, c, h, w)

            ssim_loss *= self.ssim_scaling_set

            self.log("train/loss_inverse", loss_inverse, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/ssim_loss", ssim_loss, on_step=True, on_epoch=True, prog_bar=True)

            if batch_idx == 0:
                self.logger.log_image('train/estimate_inverse', np.split(inverse_image[0].abs()/inverse_image[0].abs().max(),lambda_image.shape[1], 0))
            loss += loss_inverse
            loss += ssim_loss


        if self.pass_all_data:
            estimate_full = self.pass_through_model(undersampled, initial_mask)

            image_full = root_sum_of_squares(ifft_2d_img(estimate_full), coil_dim=2)/scaling_factor

            b, c, h, w = lambda_image.shape
            lambda_image = lambda_image.reshape(b * c, 1, h, w)
            image_full = image_full.reshape(b * c, 1, h, w)
            ssim_loss_full = torch.tensor(1, device=self.device) - ssim(lambda_image, image_full, data_range=(0, 1))
            
            if self.pass_inverse_data:
                assert inverse_image is not None, "should exist!"
                inverse_image = inverse_image.reshape(b * c, 1, h, w)
                ssim_loss_full_inverse = torch.tensor(1, device=self.device) - ssim(inverse_image, image_full, data_range=(0, 1))
                inverse_image.reshape(b, c, h, w)
                loss += ssim_loss_full_inverse * self.ssim_scaling_inverse_full

            lambda_image = lambda_image.reshape(b, c, h, w)
            image_full = image_full.reshape(b, c, h, w)
            loss += ssim_loss_full * self.ssim_scaling_full

        
        self.log("train/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_lambda", loss_lambda, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            with torch.no_grad():
                lambda_image = lambda_image.detach().cpu()
                wandb_logger = self.logger
                initial_mask = initial_mask[0, :, 0, :, :]
                lambda_set = lambda_set[0, :, 0, : ,:]
                inverse_set = inverse_set[0, :, 0, : ,:]
                wandb_logger.log_image('train/omega_lambda', np.split(lambda_set.cpu().detach().numpy(), lambda_set.shape[0], 0))
                wandb_logger.log_image('train/omega_(1-lambda)', np.split(inverse_set.cpu().detach().numpy(), inverse_set.shape[0], 0))
                wandb_logger.log_image('train/estimate_lambda', np.split(lambda_image[0].abs()/lambda_image[0].abs().max(),lambda_image.shape[1], 0))
                wandb_logger.log_image('train/initial_mask', np.split(initial_mask.cpu().detach().numpy(), lambda_set.shape[0], 0))

                probability = [torch.sigmoid(sampling_weights * self.sigmoid_slope_1) for sampling_weights in self.sampling_weights]
                R_value = self.norm_R(self.R_value)
                for i in range(len(R_value)):
                    self.log(f'train/R_{self.contrast_order[i]}', R_value[i], on_epoch=True)

                probability = self.norm_prob(probability, R_value, mask_center=True)
                probability = torch.stack(probability, dim=0)
                wandb_logger.log_image('train/probability', np.split(probability.abs(), lambda_image.shape[1], 0))
        return loss



    def validation_step(self, batch, batch_idx):
        under = batch['input']

        fs_k_space = batch['fs_k_space']
        initial_mask = (under != 0).to(torch.float32)

        nbatch, contrast, coil, h, w = under.shape
        
        mask_lambda, mask_inverse = self.split_into_lambda_loss_sets(initial_mask, nbatch)

        mask_inverse_w_acs = mask_inverse.clone()
        mask_lambda_wo_acs = mask_lambda.clone()
        mask_inverse_w_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 1
        mask_lambda_wo_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 0

        estimate_lambda = self.pass_through_model(under, mask_lambda)
        estimate_inverse = self.pass_through_model(under, mask_inverse_w_acs)
        estimate_full = self.pass_through_model(under, initial_mask)

        loss_inverse = self.loss_func(
                torch.view_as_real(estimate_inverse*mask_lambda_wo_acs), 
                torch.view_as_real(under * mask_lambda_wo_acs)
                ) 
        loss_lambda = self.loss_func(
                torch.view_as_real(estimate_lambda*mask_inverse), 
                torch.view_as_real(under * mask_inverse)
                ) 
        self.log("val/val_loss_inverse", loss_inverse, on_epoch=True, prog_bar=True)
        self.log("val/val_loss_lambda", loss_lambda, on_epoch=True, prog_bar=True)

        fully_sampled_img = root_sum_of_squares(ifft_2d_img(fs_k_space), coil_dim=2)
        scaling_factor = fully_sampled_img.amax((-1, -2), keepdim=True)
        fully_sampled_img /= scaling_factor

        est_lambda_img = root_sum_of_squares(ifft_2d_img(estimate_lambda), coil_dim=2)/scaling_factor
        est_inverse_img = root_sum_of_squares(ifft_2d_img(estimate_inverse), coil_dim=2)/scaling_factor
        est_full_img = root_sum_of_squares(ifft_2d_img(estimate_full), coil_dim=2)/scaling_factor
        est_lambda_img = est_lambda_img.clip(0, 1)
        est_full_img = est_full_img.clip(0, 1)
        est_inverse_img = est_inverse_img.clip(0, 1)

        wandb_logger = self.logger


        ssim_full_gt = evaluate_over_contrasts(ssim, fully_sampled_img, est_full_img)
        ssim_lambda_gt = evaluate_over_contrasts(ssim, fully_sampled_img, est_lambda_img)
        ssim_inverse_gt = evaluate_over_contrasts(ssim, fully_sampled_img, est_inverse_img)
        ssim_lambda_estimate = evaluate_over_contrasts(ssim, est_full_img, est_lambda_img)
        ssim_inverse_estimate = evaluate_over_contrasts(ssim, est_full_img, est_inverse_img)
        ssim_lambda_inverse = evaluate_over_contrasts(ssim, est_lambda_img, est_inverse_img)

        self.log("val/ssim_gt_full", ssim_full_gt, on_epoch=True)
        self.log("val/ssim_inverse_lambda", ssim_lambda_inverse, on_epoch=True)
        self.log("val/ssim_inverse_gt", ssim_inverse_gt, on_epoch=True)
        self.log("val/ssim_lambda_gt", ssim_lambda_gt, on_epoch=True)
        self.log("val/ssim_inverse_full", ssim_inverse_estimate, on_epoch=True)
        self.log("val/ssim_lambda_full", ssim_lambda_estimate, on_epoch=True)

        nmse_full_gt = evaluate_over_contrasts(nmse, fs_k_space, estimate_full)
        nmse_lambda_gt = evaluate_over_contrasts(nmse, fs_k_space, estimate_lambda)
        nmse_inverse_gt = evaluate_over_contrasts(nmse, fs_k_space, estimate_inverse)
        nmse_lambda_estimate = evaluate_over_contrasts(nmse, estimate_full, estimate_lambda)
        nmse_inverse_estimate = evaluate_over_contrasts(nmse, estimate_full, estimate_inverse)
        nmse_lambda_inverse = evaluate_over_contrasts(nmse, estimate_lambda, estimate_inverse)

        self.log("val/nmse_gt_full", nmse_full_gt, on_epoch=True)
        self.log("val/nmse_inverse_lambda", nmse_lambda_inverse, on_epoch=True)
        self.log("val/nmse_inverse_gt", nmse_inverse_gt, on_epoch=True)
        self.log("val/nmse_lambda_gt", nmse_lambda_gt, on_epoch=True)
        self.log("val/nmse_inverse_full", nmse_inverse_estimate, on_epoch=True)
        self.log("val/nmse_lambda_full", nmse_lambda_estimate, on_epoch=True)

        if batch_idx == 0:
            est_lambda_plot = est_lambda_img[0].cpu().numpy()
            est_inverse_plot = est_inverse_img[0].cpu().numpy()
            est_full_plot = est_full_img[0].cpu().numpy()
            fully_sampled_plot = fully_sampled_img[0].cpu().numpy()
            mask_lambda = mask_lambda[0, :, 0].cpu().numpy()
            mask_inverse = mask_inverse[0, :, 0].cpu().numpy()
            initial_mask = initial_mask[0, :, 0].cpu().numpy()
            est_lambda_plot /= np.max(est_lambda_plot, axis=(-1, -2), keepdims=True)
            est_inverse_plot /= np.max(est_inverse_plot, (-1, -2), keepdims=True)
            est_full_plot /= np.max(est_full_plot, (-1, -2), keepdims=True)
            fully_sampled_plot /= np.max(fully_sampled_plot, (-1, -2), keepdims=True)

            diff_est_lambda_plot = np.abs(est_lambda_plot - fully_sampled_plot)
            diff_est_inverse_plot = np.abs(est_inverse_plot - fully_sampled_plot)
            diff_est_full_plot = np.abs(est_full_plot - fully_sampled_plot)
            
            wandb_logger.log_image('val/estimate_lambda', np.split(est_lambda_plot, est_lambda_img.shape[1], 0))
            wandb_logger.log_image('val/estimate_inverse', np.split(est_inverse_plot, est_inverse_img.shape[1], 0))
            wandb_logger.log_image('val/estimate_full', np.split(est_full_plot, est_inverse_img.shape[1], 0))
            wandb_logger.log_image('val/ground_truth', np.split(fully_sampled_plot, est_inverse_img.shape[1], 0))

            wandb_logger.log_image('val/estimate_lambda_diff', np.split(np.clip(diff_est_lambda_plot*4, 0, 1), est_lambda_img.shape[1], 0))
            wandb_logger.log_image('val/estimate_inverse_diff', np.split(np.clip(diff_est_inverse_plot*4, 0, 1), est_inverse_img.shape[1], 0))
            wandb_logger.log_image('val/estimate_full_diff', np.split(np.clip(diff_est_full_plot*4, 0, 1), est_inverse_img.shape[1], 0))
            wandb_logger.log_image('val/omega_lambda', np.split(mask_lambda, mask_lambda.shape[0], 0))
            wandb_logger.log_image('val/omega_(1-lambda)', np.split(mask_inverse, mask_lambda.shape[0], 0))
            wandb_logger.log_image('val/initial_mask', np.split(initial_mask, initial_mask.shape[0], 0))



    def test_step(self, batch, batch_idx):
        undersampled = batch['input']
        k_space = batch['fs_k_space']

        inital_undersampling = undersampled != 0

        estimate_k = self.recon_model(undersampled, inital_undersampling)
        estimate_k = estimate_k * ~inital_undersampling + undersampled

        super().test_step((estimate_k, k_space, 'pass full'), batch_idx)

        initial_mask = undersampled != 0 
        nbatch = undersampled.shape[0]
        mask_lambda, mask_inverse = self.split_into_lambda_loss_sets(initial_mask, nbatch)
        b, contrast, c, h, w = undersampled.shape

        mask_inverse_w_acs = mask_inverse.clone()
        mask_lambda_wo_acs = mask_lambda.clone()
        mask_inverse_w_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 1
        mask_lambda_wo_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 0

        estimate_lambda = self.pass_through_model(undersampled, mask_lambda)
        estimate_lambda = estimate_lambda * ~initial_mask + undersampled 
        estimate_inverse = self.pass_through_model(undersampled, mask_inverse_w_acs)
        estimate_inverse = estimate_inverse * ~initial_mask + undersampled

        mean_lambda_inverse = torch.stack([estimate_lambda, estimate_inverse]).mean(0)
        mean_all_3 = torch.stack([estimate_lambda, estimate_inverse, estimate_k]).mean(0)
        mean_full_lambda = torch.stack([estimate_lambda, estimate_k]).mean(0)

        super().test_step((mean_lambda_inverse, k_space, 'pass lambda+inverse'), batch_idx)
        super().test_step((mean_all_3, k_space, 'pass all 3'), batch_idx)
        super().test_step((mean_full_lambda, k_space, 'pass all lambda+full'), batch_idx)

        lambda_img = root_sum_of_squares(ifft_2d_img(estimate_lambda), coil_dim=2) 
        inverse_img = root_sum_of_squares(ifft_2d_img(estimate_inverse), coil_dim=2) 
        full_img = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2) 

        variance_map = torch.stack([lambda_img, inverse_img, full_img]).std(0)
        wandb_logger = self.logger
        wandb_logger.log_image('test/std of 3 images', np.split(np.clip(variance_map[0].cpu(), 0, 1), variance_map.shape[1], 0))


    

    # takes an array of R values and normalizes it to the desired R value

    def norm_R(self, R) -> List[torch.Tensor]:
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


    def norm_prob(self, probability:List[torch.Tensor], cur_R:List[torch.Tensor], center_region=10, mask_center=False):
        image_shape = probability[0].shape
        if self.prob_method == 'loupe':
            self.norm_2d_probability(probability, cur_R, center_region, mask_center, image_shape)

        elif self.prob_method == 'line_loupe':
            self.norm_1d_probability(probability, cur_R, center_region, mask_center, image_shape)
        else:
            raise ValueError(f'No prob method found for {self.prob_method}')
        
        assert all((torch.isclose(probs.mean(), 1/R, atol=0.01, rtol=0) for probs, R in zip(probability, cur_R))), f'Probability should be equal to R {[prob.mean() for prob in probability]}'
        return probability

    def norm_1d_probability(self, probability, cur_R, center_region, mask_center, image_shape):
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

    def norm_2d_probability(self, probability, cur_R, center_region, mask_center, image_shape):
        center = [image_shape[0]//2, image_shape[1]//2]

        center_bb_x = slice(center[0]-center_region//2,center[0]+center_region//2)
        center_bb_y = slice(center[1]-center_region//2,center[1]+center_region//2)
            
        probability_sum = torch.zeros((len(probability), 1), device=self.device)

        # create acs mask of zeros for acs box and zeros elsewhere
        center_mask = torch.ones(image_shape, device=probability[0].device)
        if mask_center:
            center_mask[center_bb_y, center_bb_x] = 0

        for i in range(len(probability)):
            probability[i] = probability[i] * center_mask
            probability_sum[i] = probability[i].sum(dim=[-1, -2])
            
        for i in range(len(probability)):
            probability_total = image_shape[-1] * image_shape[-2]/ cur_R[i]
            if mask_center:
                probability_total -= center_region ** 2

            # we need to find cur_R * scaling_factor = R
            # scaling down the values of 1
            if probability_sum[i] > probability_total:
                scaling_factor = probability_total / probability_sum[i]
                assert scaling_factor <= 1 and scaling_factor >= 0

                probability[i] = (probability[i] * scaling_factor)

            # scaling down the complement probability (scaling down 0)
            else:
                inverse_total = image_shape[1]*image_shape[0]*(1 - 1/cur_R[i])
                inverse_sum = (image_shape[1]*image_shape[0]) - probability_sum[i] 
                if mask_center:
                    inverse_sum -= center_region**2
                scaling_factor = inverse_total / inverse_sum
                assert scaling_factor <= 1 and scaling_factor >= 0

                inv_prob = (1 - probability[i])*scaling_factor
                probability[i] = 1 - inv_prob
           
            # acs box is now ones and everything else is zeros
        if mask_center:
            for i in range(len(probability)):
                probability[i][center_bb_y, center_bb_x] = 1


    def get_mask(self, batch_size, mask_center=False, deterministic=False):
        # Calculate probability and normalize

        sampling_weights = self.sampling_weights
        assert not torch.isnan(sampling_weights).any(), "sampling weights shouldn't be nan!"
        assert sampling_weights.shape == self.image_size, "sampling weights should match the image size" 

        if 'loupe' in self.prob_method:
            probability = [torch.sigmoid(sampling_weights * self.sigmoid_slope_1) for sampling_weights in sampling_weights]
        else:
            raise TypeError('Only implemented 2d loupe')
        
        assert all((probs.min() >= 0 for probs in probability)), f'Probability should be greater than 1 but found {[prob.min() for prob in probability]}'
        assert all((probs.max() <= 1 for probs in probability)), f'Probability should be less than 1 but found {[prob.max() for prob in probability]}'

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

        sampling_mask = self.kMaxSampling(activation, R_value, self.sigmoid_slope_2)
        #if deterministic:
        #    sampling_mask = (activation > 0).to(torch.float)
        #else:
        #    sampling_mask = torch.sigmoid(activation * self.sigmoid_slope_2)
#
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
    
        return sampling_mask.unsqueeze(2)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 6000, eta_min=1e-2) 
        return optimizer

    def train_supervised_step(self, batch): 
        undersampled = batch['input']
        mask = undersampled != 0
        fully_sampled = batch['fs_k_space']

        estimate = self.pass_through_model(undersampled, mask)
        loss = self.loss_func(torch.view_as_real(estimate), torch.view_as_real(fully_sampled)) 

        self.log("train/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def kMaxSampling(self, input, R, slope) -> torch.Tensor:
        return KMaxSoftmaxFunction.apply(input, R, slope) # type: ignore

    def split_into_lambda_loss_sets(self, omega_mask, batch_size): 
        lambda_mask = self.get_mask(batch_size, mask_center=True)
        return omega_mask * lambda_mask, omega_mask * (1 - lambda_mask)

    def final_dc_step(self, undersampled, estimated, mask):
        return estimated * (1 - mask) + undersampled * mask

    def pass_through_model(self, undersampled, mask):
        estimate = self.recon_model(undersampled*mask, mask)
        estimate = self.final_dc_step(undersampled, estimate, mask)
        return estimate
