import torch.nn as nn
import numpy as np
import torch
import einops
from typing import List
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, StepLR
from pytorch_lightning.loggers import WandbLogger

from torchmetrics.image import StructuralSimilarityIndexMeasure
from ml_recon.losses import L1L2Loss
from ml_recon.utils.undersample_tools import gen_pdf_bern
from ml_recon.pl_modules.pl_ReconModel import plReconModel
from ml_recon.utils.evaluation_functions import nmse
from ml_recon.utils import ifft_2d_img, root_sum_of_squares
from ml_recon.models import LearnPartitioning, MultiContrastVarNet, TriplePathway
from ml_recon.utils.kmax_relaxation import KMaxSoftmaxFunction
from ml_recon.utils.evaluate_over_contrasts import evaluate_over_contrasts
from ml_recon.utils.undersample_tools import ssdu_gaussian_selection, scale_pdf, apply_undersampling_from_dist

class LearnedSSLLightning(plReconModel):
    def __init__(
            self, 
            learn_partitioning_config: LearnPartitioning.LearnPartitionConfig,
            varnet_config: MultiContrastVarNet.VarnetConfig,
            dual_domain_config: TriplePathway.DualDomainConifg,
            lr: float = 1e-3,
            ssim_scaling_set: float = 1e-4,
            ssim_scaling_full: float = 1e-4,
            ssim_scaling_inverse: float = 1e-4,
            lambda_scaling: float = 1, 
            image_loss_function: str = 'ssim',
            k_space_loss_function: str = 'l1l2',
            is_supervised_training: bool = False,
            is_learn_partitioning: bool = True,  
            warmup_training: bool = False
            ):
        super().__init__(contrast_order=varnet_config.contrast_order)
        self.save_hyperparameters(ignore=['recon_model', 'partition_model'])

        if is_learn_partitioning == 'learn':
            self.partition_model = LearnPartitioning.LearnPartitioning(learn_partitioning_config)

        self.recon_model = TriplePathway.TriplePathway(dual_domain_config, varnet_config)

        self.lr = lr
        self.image_scaling_lam_inv = ssim_scaling_set
        self.image_scaling_lam_full = ssim_scaling_full
        self.image_scaling_full_inv = ssim_scaling_inverse
        self.lambda_loss_scaling = lambda_scaling
        self.is_supervised_training = is_supervised_training
        self.is_training_warmup = warmup_training

        # loss function init
        self.ssim_func = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(self.device)
        self._setup_image_space_loss(image_loss_function)
        self._setup_k_space_loss(k_space_loss_function)


    def training_step(self, batch, batch_idx):
        image_loss_full_lambda = torch.tensor([0]) 
        image_loss_inverse_lambda = torch.tensor([0]) 
        image_loss_inverse_full = torch.tensor([0]) 
        k_loss_lambda = torch.tensor([0]) 
        k_loss_inverse = torch.tensor([0])

        input_mask, loss_mask = self.partition_k_space(batch)

        if isinstance(self.logger, WandbLogger):
            wandb_logger = self.logger
        else:
            wandb_logger = None
        
        
        estimates = self.recon_model.forward(batch['input'], batch['fs_k_space'], input_mask, loss_mask) 

        lambda_k = estimates['lambda_path']
        full_k = estimates['full_path']
        inverse_k = estimates['inverse_path']
        zero_filled_k = batch['input']

        zero_filled_i = root_sum_of_squares(ifft_2d_img(zero_filled_k), coil_dim=2) 
        image_scaling_factor = zero_filled_i.amax((-1, -2), keepdim=True)

        lambda_images = root_sum_of_squares(ifft_2d_img(lambda_k), coil_dim=2) 
        lambda_images /= image_scaling_factor

        k_loss_lambda = self.calculate_k_loss(lambda_k, zero_filled_k, loss_mask)
        k_loss_lambda *= self.lambda_loss_scaling

        full_images = None
        if full_k is not None:
            full_images = root_sum_of_squares(ifft_2d_img(full_k), coil_dim=2) 
            full_images /= image_scaling_factor

            image_loss_full_lambda = self.calculate_image_space_loss(full_images, lambda_images)
            image_loss_full_lambda *= self.image_scaling_lam_full
        
        inverse_images = None 
        if inverse_k is not None:
            lambda_k_wo_acs, inverse_k_wo_acs = TriplePathway.TriplePathway.create_inverted_masks(input_mask, loss_mask)
            inverse_images = root_sum_of_squares(ifft_2d_img(inverse_k), coil_dim=2) 
            inverse_images /= image_scaling_factor

            k_loss_inverse = self.calculate_k_loss(inverse_k, zero_filled_k, lambda_k_wo_acs)
            k_loss_inverse *= 1 - self.lambda_loss_scaling
            image_loss_inverse_lambda = self.calculate_image_space_loss(lambda_images, inverse_images)
            image_loss_inverse_lambda *= self.image_scaling_lam_inv

        if (inverse_images is not None) and (full_images is not None):
            image_loss_inverse_full = self.calculate_image_space_loss(full_images, inverse_images)
            image_loss_inverse_full *= self.image_scaling_full_inv
        

        loss = k_loss_inverse + k_loss_lambda + image_loss_full_lambda + image_loss_inverse_full + image_loss_inverse_lambda

        
        
        self.log("train/image_loss_inverse_lambda", image_loss_inverse_lambda, on_epoch=True, on_step=False)
        self.log("train/image_loss_full_lambda", image_loss_full_lambda, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/image_loss_inverse_full', image_loss_inverse_full, on_step=False, on_epoch=True)
        self.log("train/loss_lambda", k_loss_lambda, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_inverse", k_loss_inverse, on_step=True, on_epoch=True, prog_bar=True)

        
        # R-value
        R_value = self.partition_model.get_R()
        for i, contrast in enumerate(self.contrast_order):
            self.log(f'train/R_{contrast}', R_value[i])
        
        initial_mask = input_mask + loss_mask
        # percentage of k-space points in lambda set vs inverse set
        for i in range(len(self.learned_R_value)):
            self.log(f'train/lambda-over-inverse_{self.contrast_order[i]}', 
                     input_mask[:, i, 0, :, :].sum()/initial_mask[:, i, 0, :, :].sum(), 
                     on_epoch=True, on_step=False)

        if batch_idx == 0 and self.current_epoch % 10 == 0 and wandb_logger:
            with torch.no_grad():
                initial_mask = initial_mask[0, :, 0, :, :]
                lambda_set_plot = input_mask[0, :, 0, : ,:]
                loss_mask = loss_mask[0, :, 0, : ,:]
                wandb_logger.log_image('train/omega_lambda', self.convert_image_to_plot(lambda_set_plot))
                wandb_logger.log_image('train/omega_(1-lambda)', self.convert_image_to_plot(loss_mask))
                wandb_logger.log_image('train/estimate_lambda', self.convert_image_to_plot(lambda_images[0]/lambda_images[0].max()))
                wandb_logger.log_image('train/initial_mask', self.convert_image_to_plot(initial_mask))
                if inverse_images is not None:
                    wandb_logger.log_image('train/estimate_inverse', self.convert_image_to_plot(inverse_images[0]/inverse_images[0].max()))
                if full_images is not None:
                    wandb_logger.log_image('train/estimate_full', self.convert_image_to_plot(full_images[0]/full_images[0].max()))
                    

                probability = [torch.sigmoid(sampling_weights * self.sigmoid_slope_1) for sampling_weights in self.sampling_weights]
                probability = self.norm_prob(probability, R_value, mask_center=True)

                wandb_logger.log_image('train/probability', probability)

        return loss
    
    def partition_k_space(self, batch):
        undersampled = batch['input']
        initial_mask = batch['mask'].to(torch.float32)

        if self.is_learned_partitioning: 
            input_mask, loss_mask = self.partition_model(undersampled, initial_mask)
        else: 
            input_mask, loss_mask = self.get_masks_from_dataset(batch)
        return input_mask, loss_mask

    def convert_image_to_plot(self, image):
        return np.split(image.cpu().detach().numpy(), image.shape[0], 0)



    def validation_step(self, batch, batch_idx):
        under = batch['input']

        fs_k_space = batch['fs_k_space']
        initial_mask = batch['mask'].to(torch.float32)

        nbatch, contrast, coil, h, w = under.shape
        
        mask_lambda, mask_inverse = self.split_into_lambda_loss_sets(initial_mask, nbatch)

        mask_inverse_w_acs = mask_inverse.clone()
        mask_lambda_wo_acs = mask_lambda.clone()
        mask_inverse_w_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 1
        mask_lambda_wo_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 0

        estimate_lambda = self.pass_through_model(under, mask_lambda, fs_k_space)
        estimate_inverse = self.pass_through_model(under, mask_inverse_w_acs, fs_k_space)
        estimate_full = self.pass_through_model(under, initial_mask, fs_k_space)

        loss_inverse = self.k_space_loss(
                torch.view_as_real(under * mask_lambda_wo_acs),
                torch.view_as_real(estimate_inverse*mask_lambda_wo_acs), 
                ) 
        loss_lambda = self.k_space_loss(
                torch.view_as_real(under * mask_inverse),
                torch.view_as_real(estimate_lambda*mask_inverse),
                ) 
        self.log("val/val_loss_inverse", loss_inverse, on_epoch=True, prog_bar=True)
        self.log("val/val_loss_lambda", loss_lambda, on_epoch=True, prog_bar=True)

        fully_sampled_img = root_sum_of_squares(ifft_2d_img(fs_k_space), coil_dim=2)
        est_lambda_img = root_sum_of_squares(ifft_2d_img(estimate_lambda), coil_dim=2)
        est_inverse_img = root_sum_of_squares(ifft_2d_img(estimate_inverse), coil_dim=2)
        est_full_img = root_sum_of_squares(ifft_2d_img(estimate_full), coil_dim=2)

        scaling = fully_sampled_img.amax((-1, -2), keepdim=True)

        fully_sampled_img /= scaling
        est_lambda_img = torch.clip(est_lambda_img/scaling, 0, 1)
        est_inverse_img = torch.clip(est_inverse_img/scaling, 0, 1)
        est_full_img = torch.clip(est_full_img/scaling, 0, 1)
        

        self.log_metrics(
                fs_k_space, 
                estimate_lambda, 
                estimate_inverse, 
                estimate_full, 
                fully_sampled_img, 
                est_lambda_img, 
                est_inverse_img, 
                est_full_img
                )
    
        if batch_idx == 0 and isinstance(self.logger, WandbLogger) and self.current_epoch % 10 == 0:
            wandb_logger = self.logger
            assert isinstance(wandb_logger, WandbLogger)
            est_lambda_plot = est_lambda_img[0].cpu().numpy()
            est_full_plot = est_full_img[0].cpu().numpy()
            fully_sampled_plot = fully_sampled_img[0].cpu().numpy()
            mask_lambda = mask_lambda[0, :, 0].cpu().numpy()
            mask_inverse = mask_inverse[0, :, 0].cpu().numpy()
            initial_mask = initial_mask[0, :, 0].cpu().numpy()
            est_lambda_plot /= np.max(est_lambda_plot, axis=(-1, -2), keepdims=True)
            est_full_plot /= np.max(est_full_plot, (-1, -2), keepdims=True)
            fully_sampled_plot /= np.max(fully_sampled_plot, (-1, -2), keepdims=True)

            diff_est_lambda_plot = np.abs(est_lambda_plot - fully_sampled_plot)
            diff_est_full_plot = np.abs(est_full_plot - fully_sampled_plot)

            wandb_logger.log_image('val/estimate_lambda', np.split(est_lambda_plot, est_lambda_img.shape[1], 0))
            wandb_logger.log_image('val/estimate_full', np.split(est_full_plot, est_inverse_img.shape[1], 0))
            wandb_logger.log_image('val/ground_truth', np.split(fully_sampled_plot, est_inverse_img.shape[1], 0))

            wandb_logger.log_image('val/estimate_lambda_diff', np.split(np.clip(diff_est_lambda_plot*10, 0, 1), est_lambda_img.shape[1], 0))
            wandb_logger.log_image('val/estimate_full_diff', np.split(np.clip(diff_est_full_plot*10, 0, 1), est_inverse_img.shape[1], 0))

    def on_train_epoch_start(self):
        if self.current_epoch >= 50 and self.is_training_warmup:
            self.sampling_weights.requires_grad = True
    
    def log_metrics(
            self, 
            fs_k_space, 
            estimate_lambda, 
            estimate_inverse, 
            estimate_full, 
            fully_sampled_img, 
            est_lambda_img, 
            est_inverse_img, 
            est_full_img
            ):
        ssim_full_gt = evaluate_over_contrasts(self.ssim_func, fully_sampled_img, est_full_img)
        ssim_lambda_gt = evaluate_over_contrasts(self.ssim_func, fully_sampled_img, est_lambda_img)
        ssim_inverse_gt = evaluate_over_contrasts(self.ssim_func, fully_sampled_img, est_inverse_img)
        ssim_lambda_estimate = evaluate_over_contrasts(self.ssim_func, est_full_img, est_lambda_img)
        ssim_inverse_estimate = evaluate_over_contrasts(self.ssim_func, est_full_img, est_inverse_img)
        ssim_lambda_inverse = evaluate_over_contrasts(self.ssim_func, est_lambda_img, est_inverse_img)

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


    def test_step(self, batch, batch_index):
        undersampled = batch['input']
        k_space = batch['fs_k_space']
        mask = batch['mask'].to(torch.float32)
        estimate_k = self.pass_through_model(undersampled, mask, k_space)

        return super().test_step((estimate_k, k_space), batch_index)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        undersampled = batch['input']
        k_space = batch['fs_k_space']
        mask = batch['mask'].to(torch.float32)
        estimate_k = self.pass_through_model(undersampled, mask, k_space)

        return super().on_test_batch_end(outputs, (estimate_k, k_space), batch_idx, dataloader_idx)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1) 
        #step_lr = StepLR(optimizer, step_size=50)
        return optimizer

    def train_supervised_step(self, batch): 
        undersampled = batch['input']
        mask = batch['mask']
        fully_sampled = batch['fs_k_space']

        estimate = self.pass_through_model(undersampled, mask, fully_sampled)
        loss = self.k_space_loss(torch.view_as_real(fully_sampled), torch.view_as_real(estimate)) 

        self.log("train/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def _setup_image_space_loss(self, image_loss_function):
        if image_loss_function == 'ssim':
            self.image_loss_func = lambda targ, pred: 1 - self.ssim_func(targ, pred)
        elif image_loss_function == 'l1':
            l1_loss = torch.nn.L1Loss()
            image_loss = lambda targ, pred: l1_loss(targ, pred)
            self.image_loss_func = image_loss
        elif image_loss_function == 'l1_grad':
            l1_loss = torch.nn.L1Loss()
            grad_x = lambda targ, pred: l1_loss(targ.diff(dim=-1), pred.diff(dim=-1))
            grad_y = lambda targ, pred: l1_loss(targ.diff(dim=-2), pred.diff(dim=-2))
            image_loss = lambda targ, pred: (
                    2*l1_loss(targ, pred) + 
                    grad_x(targ, pred) + 
                    grad_y(targ, pred)
            )
            self.image_loss_func = image_loss


    def calculate_image_space_loss(self, image1, image2):
        b, c, h, w = image1.shape
        image1 = image1.view(b * c, 1, h, w)
        image2 = image2.view(b * c, 1, h, w)
        ssim_loss = self.image_loss_func(image1, image2)

        return ssim_loss

    def pass_through_inverse_path(self, undersampled, fs_k_space, lambda_set, inverse_set):
        mask_inverse_w_acs, mask_lambda_wo_acs = self._create_inverted_masks(lambda_set, inverse_set)

        estimate_inverse = self.pass_through_model(undersampled, mask_inverse_w_acs, fs_k_space)
            
            # calculate loss

        loss_inverse = self.calculate_k_loss(undersampled, mask_lambda_wo_acs, estimate_inverse)

        loss_inverse = loss_inverse * (1 - self.lambda_loss_scaling)
        return estimate_inverse,loss_inverse

    def calculate_k_loss(self, estimate, undersampled, loss_mask):
        loss_inverse = self.k_space_loss(
                    torch.view_as_real(estimate * loss_mask), 
                    torch.view_as_real(undersampled * loss_mask),
                    )
                    
        return loss_inverse


    def _setup_k_space_loss(self, k_space_loss_function):
        if k_space_loss_function == 'l1l2':
            self.k_space_loss = L1L2Loss(norm_all_k=False)
        elif k_space_loss_function == 'l1':
            self.k_space_loss = torch.nn.L1Loss()
        elif k_space_loss_function == 'l2': 
            self.k_space_loss = torch.nn.MSELoss()
        else:
            raise ValueError('No k-space loss!')

    def get_masks_from_dataset(self, batch):
        return batch['mask'], batch['loss_mask']