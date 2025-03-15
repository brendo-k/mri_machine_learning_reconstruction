import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
from typing import Literal, Union
from functools import partial

from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from ml_recon.losses import L1L2Loss
from ml_recon.pl_modules.pl_ReconModel import plReconModel
from ml_recon.utils.evaluation_functions import nmse
from ml_recon.utils import k_to_img
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
            image_loss_scaling_lam_inv: float = 1e-4,
            image_loss_scaling_lam_full: float = 1e-4,
            image_loss_scaling_full_inv: float = 1e-4,
            lambda_scaling: float = 1, 
            image_loss_function: str = 'ssim',
            k_space_loss_function: Literal['l1l2', 'l1', 'l2'] = 'l1l2',
            enable_learn_partitioning: bool = True,  
            enable_warmup_training: bool = False,
            pass_through_size: int = 10,
            use_supervised_image_loss: bool = False, 
            weight_decay: float = 0, 
            is_mask_testing: bool = True,
            mask_theshold: Union[dict, None] = None
            ):
        super().__init__(
            contrast_order=varnet_config.contrast_order,
            is_mask_testing=is_mask_testing,
            mask_threshold=mask_theshold
        )
        self.save_hyperparameters(ignore=['recon_model', 'partition_model'])

        if enable_learn_partitioning:
            self.partition_model = LearnPartitioning(learn_partitioning_config)

        self.recon_model = TriplePathway(dual_domain_config, varnet_config, pass_through_size=pass_through_size)

        self.lr = lr
        self.image_scaling_lam_inv = image_loss_scaling_lam_inv
        self.image_scaling_lam_full = image_loss_scaling_lam_full
        self.image_scaling_full_inv = image_loss_scaling_full_inv
        self.lambda_loss_scaling = lambda_scaling
        self.enable_warmup_training = enable_warmup_training
        self.enable_learn_partitioning = enable_learn_partitioning
        self.pass_through_size = pass_through_size
        self.use_superviesd_image_loss = use_supervised_image_loss
        self.weight_decay = weight_decay
        self.mask_threshold = mask_theshold
        self.test_metrics = is_mask_testing

        # loss function init
        self._setup_image_space_loss(image_loss_function)
        self._setup_k_space_loss(k_space_loss_function)
    
    
    def forward(self, k_space, mask, fs_k_space):
        estimate_k = self.recon_model.pass_through_model(k_space * mask, mask, fs_k_space)
        return estimate_k

    def on_train_epoch_start(self):
        # set epoch in train dataloader for updating new lambda masks
        if self.trainer.train_dataloader:
            self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch)
        return
    

    def training_step(self, batch, batch_idx):
        # init loss tensors (some may be unused)

        fully_sampled = batch['fs_k_space']
        undersampled_k = batch['undersampled']

        input_mask, loss_mask = self.partition_k_space(batch)
        estimates = self.recon_model.forward(
            undersampled_k,
            fully_sampled, 
            input_mask, 
            loss_mask,
            return_all=False,
        )


        loss_dict = self.calculate_loss(estimates, undersampled_k, fully_sampled, input_mask, loss_mask)
        loss: torch.Tensor = sum(loss for loss in loss_dict.values()) # type: ignore

        for key, value in loss_dict.items():
            self.log_scalar(f"train/{key}", value)

        self.log_scalar("train/loss", loss, on_step=True, prog_bar=True)
        
        if self.enable_learn_partitioning:
            self.log_R_value()
        
        initial_mask = undersampled_k != 0
        self.log_k_space_set_ratios(input_mask, initial_mask)

        return loss

    def log_scalar(self, label, metric, **kwargs):
        """
        Logs a scalar value to the logger.

        Args:
            label (str): The label for the scalar value.
            metric (float or torch.Tensor): The scalar value to log.
            **kwargs: Additional keyword arguments for the logger.
        """
        if "on_step" not in kwargs:
            kwargs['on_step'] = False
            
        self.log(label, metric, on_epoch=True, **kwargs)


    @torch.no_grad() 
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not isinstance(self.logger, WandbLogger):
            return

        # log first batch of every 10th epoch
        if batch_idx != 0 or self.current_epoch % 1 != 0:
            return 

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
        
        fully_sampled_images = self.k_to_img_scaled(fully_sampled, image_scaling).clip(0, 1)
        lambda_images = self.k_to_img_scaled(estimates['lambda_path'], image_scaling).clip(0, 1)

        wandb_logger.log_image('train/estimate_lambda', self.split_along_contrasts(lambda_images[0]), self.global_step)
        wandb_logger.log_image('train/fully_sampled', self.split_along_contrasts(fully_sampled_images[0]), self.global_step)

        if estimates['inverse_path'] is not None:
            inverse_images = self.k_to_img_scaled(estimates['inverse_path'], image_scaling).clip(0, 1)
            wandb_logger.log_image('train/estimate_inverse', self.split_along_contrasts(inverse_images[0]), self.global_step)

        if estimates['full_path'] is not None:
            full_images = self.k_to_img_scaled(estimates['full_path'], image_scaling).clip(0, 1)
            wandb_logger.log_image('train/estimate_full', self.split_along_contrasts(full_images[0]), self.global_step)

        # plot masks (first of the batch)
        initial_mask = (undersampled_k != 0)[0, :, 0, :, :]
        lambda_set_plot = input_mask[0, :, 0, : ,:]
        loss_mask = loss_mask[0, :, 0, : ,:]
        wandb_logger.log_image('train_masks/lambda_set', self.split_along_contrasts(lambda_set_plot), self.global_step)
        wandb_logger.log_image('train_masks/loss_set', self.split_along_contrasts(loss_mask), self.global_step)
        wandb_logger.log_image('train_masks/initial_mask', self.split_along_contrasts(initial_mask), self.global_step)

        # plot probability if learn partitioning
        if self.enable_learn_partitioning:
            probability = self.partition_model.get_probability_distribution()
            wandb_logger.log_image('probability', self.split_along_contrasts(probability), self.global_step)


    def validation_step(self, batch, batch_idx):
        noisy_data = batch[0]
        gt_data = batch[1]
        undersampled_k = noisy_data['undersampled']
        fully_sampled = noisy_data['fs_k_space']

        input_mask, loss_mask = self.partition_k_space(noisy_data)
        estimates = self.recon_model.forward(
            undersampled_k,
            fully_sampled, 
            input_mask, 
            loss_mask,
            return_all=False,
        )

        # log loss
        loss_dict = self.calculate_loss(estimates, undersampled_k, fully_sampled, input_mask, loss_mask)
        loss = sum(loss for loss in loss_dict.values())

        self.log_scalar(f"val_losses/loss", loss, prog_bar=True, sync_dist=True)
        for key, value in loss_dict.items():
            self.log_scalar(f"val_losses/{key}", value)
    

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        if batch_idx > 4 or self.current_epoch % 1  != 0:
            plot_images = False
        else:
            plot_images = True

        estimate_k, fully_sampled_k = self.infer_k_space(batch)
        ground_truth_image = batch[1]

        background_mask = self.get_image_background_mask(ground_truth_image)

        estimate_image = k_to_img(estimate_k, coil_dim=2)
        fully_sampled_image = k_to_img(fully_sampled_k, coil_dim=2)

        scaling_factor = ground_truth_image.amax((-1, -2), keepdim=True)

        estimate_image /= scaling_factor
        ground_truth_image /= scaling_factor
        fully_sampled_image /= scaling_factor

        estimate_image *= background_mask
        ground_truth_image *= background_mask
        fully_sampled_image *= background_mask
        
        diff_est_full_plot = (estimate_image - fully_sampled_image).abs()*10
        gt_diff = (estimate_image - ground_truth_image).abs()*10

        # log images
        if plot_images and isinstance(self.logger, WandbLogger):
            wandb_logger = self.logger
            wandb_logger.log_image(f'val_images_recons/estimate_full_{batch_idx}', self.split_along_contrasts(estimate_image[0].clip(0, 1)), self.global_step)
            wandb_logger.log_image(f'val_images_diff/diff_fs{batch_idx}', self.split_along_contrasts(diff_est_full_plot.clip(0, 1)[0]), self.global_step)
            wandb_logger.log_image(f'val_images_diff/diff_gt{batch_idx}', self.split_along_contrasts(gt_diff.clip(0, 1)[0]), self.global_step)

            # plot ground truths at the first epoch
            if self.current_epoch == 0:
                wandb_logger.log_image(f'val_images_target/ground_truth_{batch_idx}', self.split_along_contrasts(ground_truth_image[0].clip(0, 1)), self.global_step)
                wandb_logger.log_image(f'val_images_target/fully_sampled_{batch_idx}', self.split_along_contrasts(fully_sampled_image[0].clip(0, 1)), self.global_step)

         
        # log image space metrics 
        ssim_full = evaluate_over_contrasts(ssim, fully_sampled_image, estimate_image)
        nmse_full = evaluate_over_contrasts(nmse, fully_sampled_image, estimate_image)
        for i, contrast in enumerate(self.contrast_order):
            self.log(f"val_ssim/ssim_full_{contrast}", ssim_full[i], on_epoch=True, sync_dist=True)
            self.log(f"val_nmse/nmse_full_{contrast}", nmse_full[i], on_epoch=True, sync_dist=True)
        self.log(f"val/mean_nmse_full", sum(nmse_full)/len(nmse_full), on_epoch=True, sync_dist=True)
        self.log(f"val/mean_ssim_full", sum(ssim_full)/len(nmse_full), on_epoch=True, sync_dist=True)

        # log gt metrics 
        ssim_full = evaluate_over_contrasts(ssim, ground_truth_image, estimate_image)
        nmse_full = evaluate_over_contrasts(nmse, ground_truth_image, estimate_image)
        for i, contrast in enumerate(self.contrast_order):
            self.log(f"GT_val_ssim/ssim_full_{contrast}", ssim_full[i], on_epoch=True, sync_dist=True)
            self.log(f"GT_val_nmse/nmse_full_{contrast}", nmse_full[i], on_epoch=True, sync_dist=True)
        self.log(f"val/GT_mean_nmse_full", sum(nmse_full)/len(nmse_full), on_epoch=True, sync_dist=True)
        self.log(f"val/GT_mean_ssim_full", sum(ssim_full)/len(nmse_full), on_epoch=True, sync_dist=True)


    def test_step(self, batch, batch_index):
        estimate_k, fs_k = self.infer_k_space(batch)
        ground_truth_image = batch[1] # averaged or denoised ground truth
        fully_sampled_imgage = k_to_img(fs_k, coil_dim=2) # noisy, fully sampled ground truth
        

        gt_metrics = super().my_test_step((estimate_k, ground_truth_image), batch_index, 'gt')
        fs_metrics = super().my_test_step((estimate_k, fully_sampled_imgage), batch_index, 'fs')

        return {
            'gt_metrics': gt_metrics, 
            'fs_metrics': fs_metrics
        }


    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        ground_truth_image = batch[1]
        estimate_k, _ = self.infer_k_space(batch)

        return super().on_test_batch_end(outputs, (estimate_k, ground_truth_image), batch_idx, dataloader_idx)
    
    
    def infer_k_space(self, batch):
        k_space = batch[0]
        scaling_factor = batch[0]['scaling_factor']
        fully_sampled_k = k_space['fs_k_space']
        undersampled = k_space['undersampled']
        mask = k_space['mask']
        if (k_space['loss_mask'] * mask == 0).all(): # if disjoint masks, combine
            mask += k_space['loss_mask'] # combine to get original sampliing mask
        # pass inital data through model
        estimate_k = self.recon_model.pass_through_model(undersampled, mask, fully_sampled_k)
        estimate_k *= scaling_factor
        fully_sampled_k *= scaling_factor
        return estimate_k, fully_sampled_k


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        #warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1) 
        #step_lr = StepLR(optimizer, step_size=50)
        return optimizer

    

    def _setup_image_space_loss(self, image_loss_function):
        if image_loss_function == 'ssim':
            image_loss = lambda targ, pred: 1 - ssim(targ, pred, kernel_size=7) # type: ignore
        elif image_loss_function == 'l1':
            l1_loss = torch.nn.L1Loss()
            image_loss = lambda targ, pred: l1_loss(targ, pred)
        elif image_loss_function == 'l1_grad':
            l1_loss = torch.nn.L1Loss()
            grad_x = lambda targ, pred: l1_loss(targ.diff(dim=-1), pred.diff(dim=-1))
            grad_y = lambda targ, pred: l1_loss(targ.diff(dim=-2), pred.diff(dim=-2))
            image_loss = lambda targ, pred: (
                    l1_loss(targ, pred) + 
                    grad_x(targ, pred) + 
                    grad_y(targ, pred)
            )
        else: 
            raise ValueError(f"unsuported image loss function: {image_loss_function}")
        self.image_loss_func = image_loss


    def compute_image_loss(self, kspace1, kspace2, undersampled_k, loss_scaling):
        scaling_factor = k_to_img(undersampled_k).amax((-1, -2), keepdim=True)
        img_1 = k_to_img(kspace1)/scaling_factor
        img_2 = k_to_img(kspace2)/scaling_factor

        b, c, h, w = img_1.shape
        img_1 = img_1.view(b * c, 1, h, w)
        img_2 = img_2.view(b * c, 1, h, w)
        ssim_loss = self.image_loss_func(img_1, img_2)

        return ssim_loss * loss_scaling


    def calculate_k_loss(self, estimate, undersampled, loss_mask, loss_scaling):
        k_loss = self.k_space_loss(
                    torch.view_as_real(undersampled * loss_mask),
                    torch.view_as_real(estimate * loss_mask), 
                    )
        #if self.is_norm_loss:
        #    k_loss /= loss_mask.sum() # normalize based on loss mask
                    
        return k_loss * loss_scaling


    def _setup_k_space_loss(self, k_space_loss_function):
        #if self.is_norm_loss:
        #    reduce = 'sum'
        #else:
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
        """
        Logs the R value for each contrast in the contrast order.

        The R value is obtained from the partition model and logged for each contrast.
        """
        R_value = self.partition_model.get_R()
        for i, contrast in enumerate(self.contrast_order):
            self.log(f'sampling_metrics/R_{contrast}', R_value[i])

    def log_k_space_set_ratios(self, input_mask, initial_mask):
        for i, contrast in enumerate(self.contrast_order):
            self.log(f'sampling_metrics/lambda-over-inverse_{contrast}', 
                     input_mask[:, i, 0, :, :].sum()/initial_mask[:, i, 0, :, :].sum(), 
                     on_epoch=True, on_step=False)

                     
    def calculate_inverse_k_loss(self, input_mask, loss_mask, inverse_k, undersampled_k):
        _, lambda_k_wo_acs = TriplePathway.create_inverted_masks(input_mask, loss_mask, self.pass_through_size)
        k_loss_inverse = self.calculate_k_loss(inverse_k, undersampled_k, lambda_k_wo_acs, (1 - self.lambda_loss_scaling))
        return k_loss_inverse

    
    def partition_k_space(self, batch):
        # compute either learned or heuristic partioning masks
        if self.enable_learn_partitioning: 
            assert (batch['mask'] * batch['loss_mask'] == 0).all()
            initial_mask = batch['mask'] + batch['loss_mask']
            input_mask, loss_mask = self.partition_model(initial_mask)
        else: 
            input_mask, loss_mask = batch['mask'], batch['loss_mask']

        return input_mask, loss_mask

    def split_along_contrasts(self, image):
        return np.split(image.cpu().detach().numpy(), image.shape[0], 0)

    def k_to_img_scaled(self, k_space, scaling_factor):
        return k_to_img(k_space) / scaling_factor

    def calculate_loss(self, estimates, undersampled_k, fully_sampled, input_mask, dc_mask):
        """
        Calculate the loss for different pathways in the reconstruction process.
        Args:
            estimates (dict): Dictionary containing estimated k-space data from different paths.
                - 'lambda_path': Estimated k-space from the lambda path.
                - 'full_path': Estimated k-space from the full path.
                - 'inverse_path': Estimated k-space from the inverse path.
            undersampled_k (torch.Tensor): The undersampled k-space data.
            fully_sampled (torch.Tensor): The fully sampled k-space data.
            input_mask (torch.Tensor): The input mask for the k-space data.
            dc_mask (torch.Tensor): The data consistency mask.
        Returns:
            dict: A dictionary containing the calculated losses for different pathways if available.
            Only the losses for the available pathways are returned.
                - 'k_loss_lambda': Loss for the lambda path in k-space.
                - 'image_loss_full_lambda': Image loss for the full lambda path.
                - 'k_loss_inverse': Loss for the inverse path in k-space.
                - 'image_loss_inverse_lambda': Image loss for the inverse lambda path.
                - 'image_loss_inverse_full': Image loss for the inverse full path.
                - 'image_loss: Image loss for the lambda path vs fully sampled.'
        """
        # estimated k-space from different paths
        lambda_k = estimates['lambda_path']
        full_k = estimates['full_path']
        inverse_k = estimates['inverse_path']

        lam_full_scaling, lam_inv_scaling, inv_full_scaling = self.get_image_space_scaling_factors()

        loss_dict = {}
        if self.use_superviesd_image_loss:
            target_img = k_to_img(fully_sampled, coil_dim=2)
            lambda_img = k_to_img(lambda_k, coil_dim=2) 
            ssim_val = ssim(target_img, lambda_img, data_range=(0, target_img.max().item()))
            assert isinstance(ssim_val, torch.Tensor)
            loss_dict['image_loss'] = 1 - ssim_val

        loss_dict['k_loss_lambda'] = self.calculate_k_loss(lambda_k, fully_sampled, dc_mask, self.lambda_loss_scaling)
        # calculate full lambda image loss pathway
        if full_k is not None:
            loss_dict["image_loss_full_lambda"] = self.compute_image_loss(
            full_k, 
            lambda_k, 
            undersampled_k, 
            lam_full_scaling
            )

        # calculate inverse lambda image and k-space loss pathway 
        if inverse_k is not None:
            # k space loss
            loss_dict["k_loss_inverse"] = self.calculate_inverse_k_loss(input_mask, dc_mask, inverse_k, undersampled_k)
            # image space loss
            loss_dict["image_loss_inverse_lambda"] = self.compute_image_loss(
            lambda_k, 
            inverse_k,
            undersampled_k,
            lam_inv_scaling
            )
        # calculate inverse full loss image pathway
        if (inverse_k is not None) and (full_k is not None):
            loss_dict["image_loss_inverse_full"] = self.compute_image_loss(
            full_k, 
            inverse_k,
            undersampled_k,
            inv_full_scaling
            )
            
            
        return loss_dict

    def get_image_space_scaling_factors(self):
        if self.enable_warmup_training and self.current_epoch < 10:
            scaling_factor = self.current_epoch / 10
        else: 
            scaling_factor = 1
        
        return (
            scaling_factor * self.image_scaling_lam_full, 
            scaling_factor * self.image_scaling_lam_inv, 
            scaling_factor * self.image_scaling_full_inv
            )
        
