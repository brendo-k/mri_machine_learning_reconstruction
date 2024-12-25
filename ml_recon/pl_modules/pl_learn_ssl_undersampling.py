import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, StepLR
from pytorch_lightning.loggers import WandbLogger

from torchmetrics.image import StructuralSimilarityIndexMeasure

from ml_recon.losses import L1L2Loss
from ml_recon.pl_modules.pl_ReconModel import plReconModel
from ml_recon.utils.evaluation_functions import nmse
from ml_recon.utils import ifft_2d_img, root_sum_of_squares, k_to_img
from ml_recon.models.LearnPartitioning import LearnPartitioning,  VarnetConfig, LearnPartitionConfig
from ml_recon.models.TriplePathway import TriplePathway, DualDomainConifg
from ml_recon.utils.evaluate_over_contrasts import evaluate_over_contrasts

class LearnedSSLLightning(plReconModel):
    def __init__(
            self, 
            learn_partitioning_config: LearnPartitionConfig,
            varnet_config: VarnetConfig,
            dual_domain_config: DualDomainConifg,
            lr: float = 1e-3,
            ssim_scaling_set: float = 1e-4,
            ssim_scaling_full: float = 1e-4,
            ssim_scaling_inverse: float = 1e-4,
            lambda_scaling: float = 1, 
            image_loss_function: str = 'ssim',
            k_space_loss_function: str = 'l1l2',
            is_learn_partitioning: bool = True,  
            warmup_training: bool = False,
            is_norm_loss: bool = False,
            ):
        super().__init__(contrast_order=varnet_config.contrast_order)
        self.save_hyperparameters(ignore=['recon_model', 'partition_model'])

        if is_learn_partitioning:
            self.partition_model = LearnPartitioning(learn_partitioning_config)

        self.recon_model = TriplePathway(dual_domain_config, varnet_config)

        self.lr = lr
        self.image_scaling_lam_inv = ssim_scaling_set
        self.image_scaling_lam_full = ssim_scaling_full
        self.image_scaling_full_inv = ssim_scaling_inverse
        self.lambda_loss_scaling = lambda_scaling
        self.is_training_warmup = warmup_training
        self.is_learn_partitioning = is_learn_partitioning
        self.is_norm_loss = is_norm_loss

        # loss function init
        self.ssim_func = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(self.device)
        self._setup_image_space_loss(image_loss_function)
        self._setup_k_space_loss(k_space_loss_function)


    def training_step(self, batch, batch_idx):
        # init loss tensors (some may be unused)
        image_loss_full_lambda = torch.tensor([0.0], device=self.device) 
        image_loss_inverse_lambda = torch.tensor([0.0], device=self.device) 
        image_loss_inverse_full = torch.tensor([0.0], device=self.device) 
        k_loss_lambda = torch.tensor([0.0], device=self.device) 
        k_loss_inverse = torch.tensor([0.0], device=self.device)

        fully_sampled = batch['fs_k_space']
        undersampled_k = batch['undersampled']

        input_mask, loss_mask = self.partition_k_space(batch)
        estimates = self.recon_model.forward(
            undersampled_k,
            fully_sampled, 
            input_mask, 
            loss_mask,
            return_all=False
        )

        # estimated k-space from different paths
        lambda_k = estimates['lambda_path']
        full_k = estimates['full_path']
        inverse_k = estimates['inverse_path']

        k_loss_lambda = self.calculate_k_loss(lambda_k, fully_sampled, loss_mask, self.lambda_loss_scaling)

        # calculate full lambda image loss pathway
        if full_k is not None:
            image_loss_full_lambda = self.compute_image_loss(
                full_k, 
                lambda_k, 
                undersampled_k, 
                self.image_scaling_lam_full
                )

        # calculate inverse lambda image and k-space loss pathway 
        if inverse_k is not None:
            # k space loss
            k_loss_inverse = self.calculate_inverse_k_loss(input_mask, loss_mask, inverse_k, undersampled_k)
            # image space loss
            image_loss_inverse_lambda = self.compute_image_loss(
                lambda_k, 
                inverse_k,
                undersampled_k,
                self.image_scaling_lam_inv
                )
        # calculate inverse full loss image pathway
        if (inverse_k is not None) and (full_k is not None):
            image_loss_inverse_full = self.compute_image_loss(
                full_k, 
                inverse_k,
                undersampled_k,
                self.image_scaling_full_inv
                )
        

        loss = k_loss_inverse + k_loss_lambda + image_loss_full_lambda + image_loss_inverse_full + image_loss_inverse_lambda


        if image_loss_inverse_lambda.item() != 0:
            self.log_scalar("train/image_loss_inverse_lambda", image_loss_inverse_lambda)
        if image_loss_full_lambda.item() != 0:
            self.log_scalar("train/image_loss_full_lambda", image_loss_full_lambda)
        if image_loss_inverse_full.item() != 0:
            self.log_scalar('train/image_loss_inverse_full', image_loss_inverse_full)
        if k_loss_inverse.item() != 0:
            self.log_scalar("train/loss_inverse", k_loss_inverse)

        self.log_scalar("train/loss_lambda", k_loss_lambda, on_step=True, prog_bar=True)
        self.log_scalar("train/loss", loss, on_step=True, prog_bar=True)
        
        if self.is_learn_partitioning:
            self.log_R_value()
        
        initial_mask = undersampled_k != 0
        self.log_k_space_set_ratios(input_mask, initial_mask)

        return loss

    def log_scalar(self, label, metric, **kwargs):
        if "on_step" not in kwargs:
            kwargs['on_step'] = False
            
        self.log(label, metric, on_epoch=True, **kwargs)


    @torch.no_grad() 
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not isinstance(self.logger, WandbLogger):
            return

        # log first batch of every 10th epoch
        if batch_idx != 0 or self.current_epoch % 10 != 0:
            return 
        fully_sampled = batch['fs_k_space']
        undersampled_k = batch['undersampled']

        input_mask, loss_mask = self.partition_k_space(batch)
        estimates = self.recon_model.forward(
            undersampled_k,
            fully_sampled, 
            input_mask, 
            loss_mask,
            return_all=False
        )


        wandb_logger = self.logger
        fully_sampled = batch['fs_k_space']
        undersampled_k = batch['undersampled']

        input_mask, loss_mask = self.partition_k_space(batch)
        estimates = self.recon_model.forward(
            undersampled_k,
            fully_sampled, 
            input_mask, 
            loss_mask,
            return_all=False
        )

        # plot images (first of the batch)
        image_scaling = k_to_img(fully_sampled).amax((-1, -2), keepdim=True)
        
        fully_sampled_images = self.k_to_img_scaled(fully_sampled, image_scaling)
        lambda_images = self.k_to_img_scaled(estimates['lambda_path'], image_scaling)

        wandb_logger.log_image('train/estimate_lambda', self.split_along_contrasts(lambda_images[0]))
        wandb_logger.log_image('train/fully_sampled', self.split_along_contrasts(fully_sampled_images[0]))

        if estimates['inverse_path'] is not None:
            inverse_images = self.k_to_img_scaled(estimates['inverse_path'], image_scaling)
            wandb_logger.log_image('train/estimate_inverse', self.split_along_contrasts(inverse_images[0]))

        if estimates['full_path'] is not None:
            full_images = self.k_to_img_scaled(estimates['full_path'], image_scaling)
            wandb_logger.log_image('train/estimate_full', self.split_along_contrasts(full_images[0]))

        # plot masks (first of the batch)
        initial_mask = (undersampled_k != 0)[0, :, 0, :, :]
        lambda_set_plot = input_mask[0, :, 0, : ,:]
        loss_mask = loss_mask[0, :, 0, : ,:]
        wandb_logger.log_image('train/lambda_set', self.split_along_contrasts(lambda_set_plot))
        wandb_logger.log_image('train/loss_set', self.split_along_contrasts(loss_mask))
        wandb_logger.log_image('train/initial_mask', self.split_along_contrasts(initial_mask))

        # plot probability if learn partitioning
        if self.is_learn_partitioning:
            probability = self.partition_model.get_norm_probability()
            wandb_logger.log_image('train/probability', self.split_along_contrasts(probability))


    def validation_step(self, batch, batch_idx):
        under = batch['undersampled']
        fs_k_space = batch['fs_k_space']

        lambda_set, loss_set = self.partition_k_space(batch)
        estimates = self.recon_model(
            under, 
            fs_k_space, 
            lambda_set, 
            loss_set, 
            return_all=True
        )

        estimate_lambda = estimates['lambda_path']
        estimate_full = estimates['full_path']
        estimate_inverse = estimates['inverse_path']

        loss_lambda = self.calculate_k_loss(estimate_lambda, fs_k_space, loss_set, self.lambda_loss_scaling)
        loss_inverse = self.calculate_inverse_k_loss(lambda_set, loss_set, estimate_inverse, under)

        self.log_scalar("val/val_loss_inverse", loss_inverse, prog_bar=True)
        self.log_scalar("val/val_loss_lambda", loss_lambda, prog_bar=True)
        
        self.log_image_metrics(fs_k_space, estimate_lambda, estimate_inverse, estimate_full)
    

    def on_train_epoch_start(self):
        if self.current_epoch >= 50 and self.is_training_warmup:
            self.sampling_weights.requires_grad = True
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        if batch_idx != 0 or not isinstance(self.logger, WandbLogger) or self.current_epoch % 10 != 0:
            return

        under = batch['undersampled']
        fs_k_space = batch['fs_k_space']

        lambda_set, loss_set = self.partition_k_space(batch)
        estimates = self.recon_model(under, fs_k_space, lambda_set, loss_set, return_all=True)

        estimate_lambda = estimates['lambda_path']
        estimate_full = estimates['full_path']
        estimate_inverse = estimates['inverse_path']

        image_scaling = k_to_img(fs_k_space).amax((-1, -2), keepdim=True)
        fully_sampling_img = self.k_to_img_scaled(fs_k_space, image_scaling)
        lambda_path_img = self.k_to_img_scaled(estimate_lambda, image_scaling)
        inverse_path_img = self.k_to_img_scaled(estimate_inverse, image_scaling)
        full_path_img = self.k_to_img_scaled(estimate_full, image_scaling)
        
        diff_lambda_fs = (lambda_path_img - fully_sampling_img).abs()*10
        diff_est_full_plot = (full_path_img - fully_sampling_img).abs()*10


        wandb_logger = self.logger
        wandb_logger.log_image('val/estimate_lambda', self.split_along_contrasts(lambda_path_img[0]))
        wandb_logger.log_image('val/estimate_inverse', self.split_along_contrasts(inverse_path_img[0]))
        wandb_logger.log_image('val/estimate_full', self.split_along_contrasts(full_path_img[0]))
        wandb_logger.log_image('val/ground_truth', self.split_along_contrasts(fully_sampling_img[0]))

        wandb_logger.log_image('val/estimate_lambda_diff', self.split_along_contrasts(diff_lambda_fs.clip(0, 1)[0]))
        wandb_logger.log_image('val/estimate_full_diff', self.split_along_contrasts(diff_est_full_plot.clip(0, 1)[0]))

        lambda_set_plot = lambda_set[0, :, 0, : ,:]
        loss_mask = loss_set[0, :, 0, : ,:]
        wandb_logger.log_image('val/lambda_set', self.split_along_contrasts(lambda_set_plot.clip(0, 1)))
        wandb_logger.log_image('val/loss_set', self.split_along_contrasts(loss_mask.clip(0, 1)))
    
    
    def log_image_metrics(self, fs_k_space, estimate_lambda, estimate_inverse=None, estimate_full=None):
        """
        Logs SSIM and NMSE metrics for different reconstructions against the ground truth and each other.

        Args:
            fs_k_space (torch.Tensor): Fully sampled k-space data (ground truth).
            estimate_lambda (torch.Tensor): Lambda-based estimated k-space data.
            estimate_inverse (torch.Tensor, optional): Inverse estimated k-space data. Defaults to None.
            estimate_full (torch.Tensor, optional): Fully estimated k-space data. Defaults to None.
        """
        # Get scaling factor
        image_scaling = k_to_img(fs_k_space).amax(dim=(-1, -2), keepdim=True)

        # convert to images and clip 0, 1
        fully_sampled_img = self.k_to_img_scaled(fs_k_space, image_scaling)
        est_lambda_img = self.k_to_img_scaled(estimate_lambda, image_scaling)
        est_inverse_img = self.k_to_img_scaled(estimate_inverse, image_scaling)
        est_full_img = self.k_to_img_scaled(estimate_full, image_scaling)

        # Helper function to log SSIM metrics
        def log_ssim(label, img1, img2):
            if img1 is not None and img2 is not None:
                ssim = evaluate_over_contrasts(self.ssim_func, img1, img2)
                self.log(f"val/ssim_{label}", ssim, on_epoch=True)

        # Log SSIM metrics
        log_ssim("gt_full", fully_sampled_img, est_full_img)
        log_ssim("gt_inverse", fully_sampled_img, est_inverse_img)
        log_ssim("gt_lambda", fully_sampled_img, est_lambda_img)

        # Helper function to log NMSE metrics
        def log_nmse(label, img1, img2):
            if img1 is not None and img2 is not None:
                nmse_val = evaluate_over_contrasts(nmse, img1, img2)
                self.log(f"val/nmse_{label}", nmse_val, on_epoch=True)

        # Log NMSE metrics
        log_nmse("gt_full", fully_sampled_img, est_full_img)
        log_nmse("gt_inverse", fully_sampled_img, est_inverse_img)
        log_nmse("gt_lambda", fully_sampled_img, est_lambda_img)


    def k_to_img_scaled(self, estimate_lambda, image_scaling):
        est_lambda_img = torch.clip(k_to_img(estimate_lambda) / image_scaling, 0, 1)
        return est_lambda_img


    def test_step(self, batch, batch_index):
        k_space = batch[0]
        ground_truth_image = batch[1]
        fully_sampled_k = batch['fs_k_space']
        undersampled = batch['undersampled']
        mask = batch['mask']
        if (batch['loss_mask'] * mask == 0).all(): # if disjoint masks, combine
            mask += batch['loss_mask'] # combine to get original sampliing mask
        # pass inital data through model
        estimate_k = self.recon_model.pass_through_model(undersampled, mask, fully_sampled_k)

        return super().test_step((estimate_k, ground_truth_image), batch_index)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        fully_sampled_k = batch['fs_k_space']
        undersampled = batch['undersampled']
        mask = batch['mask']
        if (batch['loss_mask'] * mask == 0).all(): # if disjoint masks, combine
            mask += batch['loss_mask'] # combine to get original sampliing mask
        # pass inital data through the model
        estimate_k = self.recon_model.pass_through_model(undersampled, mask, fully_sampled_k)

        return super().on_test_batch_end(outputs, (estimate_k, fully_sampled_k), batch_idx, dataloader_idx)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1) 
        #step_lr = StepLR(optimizer, step_size=50)
        return optimizer


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


    def compute_image_loss(self, kspace1, kspace2, undersampled_k, loss_scaling):
        scaling_factor = k_to_img(undersampled_k).amax((-1, -2), keepdim=True)
        kspace1_img = k_to_img(kspace1)/scaling_factor
        kspace2_img = k_to_img(kspace2)/scaling_factor

        b, c, h, w = kspace1_img.shape
        kspace1_img = kspace1_img.view(b * c, 1, h, w)
        kspace2_img = kspace2_img.view(b * c, 1, h, w)
        ssim_loss = self.image_loss_func(kspace1_img, kspace2_img)

        return ssim_loss * loss_scaling


    def calculate_k_loss(self, estimate, undersampled, loss_mask, loss_scaling):
        k_loss = self.k_space_loss(
                    torch.view_as_real(undersampled * loss_mask),
                    torch.view_as_real(estimate * loss_mask), 
                    )
        if self.is_norm_loss:
            k_loss /= loss_mask.sum() # normalize based on loss mask
                    
        return k_loss * loss_scaling


    def _setup_k_space_loss(self, k_space_loss_function):
        if self.is_norm_loss:
            reduce = 'sum'
        else:
            reduce = 'mean'

        if k_space_loss_function == 'l1l2':
            self.k_space_loss = L1L2Loss(norm_all_k=False, reduce=reduce)
        elif k_space_loss_function == 'l1':
            self.k_space_loss = torch.nn.L1Loss(reduction=reduce)
        elif k_space_loss_function == 'l2': 
            self.k_space_loss = torch.nn.MSELoss(reduction=reduce)
        else:
            raise ValueError('No k-space loss!')

    def log_R_value(self):
        R_value = self.partition_model.get_R()
        for i, contrast in enumerate(self.contrast_order):
            self.log(f'train/R_{contrast}', R_value[i])

    def log_k_space_set_ratios(self, input_mask, initial_mask):
        for i, contrast in enumerate(self.contrast_order):
            self.log(f'train/lambda-over-inverse_{contrast}', 
                     input_mask[:, i, 0, :, :].sum()/initial_mask[:, i, 0, :, :].sum(), 
                     on_epoch=True, on_step=False)

                     
    def k_to_img(self, k_space):
        return root_sum_of_squares(ifft_2d_img(k_space), coil_dim=2)


    def calculate_inverse_k_loss(self, input_mask, loss_mask, inverse_k, undersampled_k):
        _, lambda_k_wo_acs = TriplePathway.create_inverted_masks(input_mask, loss_mask)
        k_loss_inverse = self.calculate_k_loss(inverse_k, undersampled_k, lambda_k_wo_acs, (1 - self.lambda_loss_scaling))
        return k_loss_inverse
    
    def partition_k_space(self, batch):
        # compute either learned or heuristic partioning masks
        if self.is_learn_partitioning: 
            assert (batch['mask'] * batch['loss_mask'] == 0).all()
            initial_mask = batch['mask'] + batch['loss_mask']
            input_mask, loss_mask = self.partition_model(initial_mask)
        else: 
            input_mask, loss_mask = batch['mask'], batch['loss_mask']

        return input_mask, loss_mask

    def split_along_contrasts(self, image):
        return np.split(image.cpu().detach().numpy(), image.shape[0], 0)
