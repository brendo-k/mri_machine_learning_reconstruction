import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
from typing import Literal

from torchmetrics.image import StructuralSimilarityIndexMeasure
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
            ssim_scaling_set: float = 1e-4,
            ssim_scaling_full: float = 1e-4,
            ssim_scaling_inverse: float = 1e-4,
            lambda_scaling: float = 1, 
            image_loss_function: str = 'ssim',
            k_space_loss_function: Literal['l1l2', 'l1', 'l2'] = 'l1l2',
            enable_learn_partitioning: bool = True,  
            enable_warmup_training: bool = False,
            pass_through_size: int = 10,
            use_supervised_image_loss: bool = False, 
            weight_decay: float = 0
            ):
        super().__init__(contrast_order=varnet_config.contrast_order)
        self.save_hyperparameters(ignore=['recon_model', 'partition_model'])

        if enable_learn_partitioning:
            self.partition_model = LearnPartitioning(learn_partitioning_config)
            #self.partition_model_compiled = torch.compile(self.partition_model)

        self.recon_model = TriplePathway(dual_domain_config, varnet_config, pass_through_size=pass_through_size)

        self.lr = lr
        self.image_scaling_lam_inv = ssim_scaling_set
        self.image_scaling_lam_full = ssim_scaling_full
        self.image_scaling_full_inv = ssim_scaling_inverse
        self.lambda_loss_scaling = lambda_scaling
        self.enable_warmup_training = enable_warmup_training
        self.enable_learn_partitioning = enable_learn_partitioning
        self.pass_through_size = pass_through_size
        self.use_superviesd_image_loss = use_supervised_image_loss
        self.weight_decay = weight_decay
        

        # loss function init
        self.ssim_func = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(self.device)
        self._setup_image_space_loss(image_loss_function)
        self._setup_k_space_loss(k_space_loss_function)


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
        if batch_idx != 0 or self.current_epoch % 10 != 0:
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

        wandb_logger.log_image('train/estimate_lambda', self.split_along_contrasts(lambda_images[0]))
        wandb_logger.log_image('train/fully_sampled', self.split_along_contrasts(fully_sampled_images[0]))

        if estimates['inverse_path'] is not None:
            inverse_images = self.k_to_img_scaled(estimates['inverse_path'], image_scaling).clip(0, 1)
            wandb_logger.log_image('train/estimate_inverse', self.split_along_contrasts(inverse_images[0]))

        if estimates['full_path'] is not None:
            full_images = self.k_to_img_scaled(estimates['full_path'], image_scaling).clip(0, 1)
            wandb_logger.log_image('train/estimate_full', self.split_along_contrasts(full_images[0]))

        # plot masks (first of the batch)
        initial_mask = (undersampled_k != 0)[0, :, 0, :, :]
        lambda_set_plot = input_mask[0, :, 0, : ,:]
        loss_mask = loss_mask[0, :, 0, : ,:]
        wandb_logger.log_image('train/lambda_set', self.split_along_contrasts(lambda_set_plot))
        wandb_logger.log_image('train/loss_set', self.split_along_contrasts(loss_mask))
        wandb_logger.log_image('train/initial_mask', self.split_along_contrasts(initial_mask))

        # plot probability if learn partitioning
        if self.enable_learn_partitioning:
            probability = self.partition_model.get_norm_probability()
            wandb_logger.log_image('train/probability', self.split_along_contrasts(probability))


    def validation_step(self, batch, batch_idx):
        undersampled_k = batch['undersampled']
        fully_sampled = batch['fs_k_space']

        input_mask, loss_mask = self.partition_k_space(batch)
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

        self.log_scalar(f"val/loss", loss, prog_bar=True, sync_dist=True)
        for key, value in loss_dict.items():
            self.log_scalar(f"val/{key}", value)
    

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        if batch_idx != 0 or self.current_epoch % 10 != 0:
            plot_images = False
        else:
            plot_images = True

        under = batch['undersampled']
        fs_k_space = batch['fs_k_space']

        lambda_set, loss_set = self.partition_k_space(batch)
        mask = lambda_set + loss_set
        
        # get recon estimates 
        estimate_full = self.recon_model.pass_through_model(under, mask, fs_k_space) 
        estimate_lambda = self.recon_model.pass_through_model(under, lambda_set, fs_k_space) 

        # ifft and root sum of squares. All iamges scaled by the max of fully sampled images.
        image_scaling = k_to_img(fs_k_space).amax((-1, -2), keepdim=True)
        fully_sampling_img = self.k_to_img_scaled(fs_k_space, image_scaling)
        estimate_full_img = self.k_to_img_scaled(estimate_full, image_scaling)
        estimate_lambda_img = self.k_to_img_scaled(estimate_lambda, image_scaling)

        # Mask background
        mask = fully_sampling_img > 0.09
        estimate_lambda_img *= mask
        estimate_full_img *= mask
        fully_sampling_img *= mask
        
        diff_est_full_plot = (estimate_full_img - fully_sampling_img).abs()*10
        diff_est_lambda_plot = (estimate_lambda_img - fully_sampling_img).abs()*10

        # log images
        if plot_images and isinstance(self.logger, WandbLogger):
            wandb_logger = self.logger
            wandb_logger.log_image('val/estimate_full', self.split_along_contrasts(estimate_full_img[0].clip(0, 1)))
            wandb_logger.log_image('val/estimate_lambda', self.split_along_contrasts(estimate_lambda_img[0].clip(0, 1)))
            wandb_logger.log_image('val/ground_truth', self.split_along_contrasts(fully_sampling_img[0].clip(0, 1)))
            wandb_logger.log_image('val/estimate_full_diff', self.split_along_contrasts(diff_est_full_plot.clip(0, 1)[0]))
            wandb_logger.log_image('val/estimate_lambda_diff', self.split_along_contrasts(diff_est_lambda_plot.clip(0, 1)[0]))

            lambda_set_plot = lambda_set[0, :, 0, : ,:]
            loss_mask = loss_set[0, :, 0, : ,:]
            wandb_logger.log_image('val/lambda_set', self.split_along_contrasts(lambda_set_plot.clip(0, 1)))
            wandb_logger.log_image('val/loss_set', self.split_along_contrasts(loss_mask.clip(0, 1)))

        
        # log image space metrics 
        ssim_full = evaluate_over_contrasts(self.ssim_func, fully_sampling_img, estimate_full_img)
        nmse_full = evaluate_over_contrasts(nmse, fully_sampling_img, estimate_full_img)
        ssim_lambda = evaluate_over_contrasts(self.ssim_func, fully_sampling_img, estimate_lambda_img)
        nmse_lambda = evaluate_over_contrasts(nmse, fully_sampling_img, estimate_lambda_img)
        for i, contrast in enumerate(self.contrast_order):
            self.log(f"val/ssim_full_{contrast}", ssim_full[i], on_epoch=True, sync_dist=True)
            self.log(f"val/nmse_full_{contrast}", nmse_full[i], on_epoch=True, sync_dist=True)
            self.log(f"val/ssim_lambda_{contrast}", ssim_lambda[i], on_epoch=True, sync_dist=True)
            self.log(f"val/nmse_lambda_{contrast}", nmse_lambda[i], on_epoch=True, sync_dist=True)
        self.log(f"val/mean_nmse_full", sum(nmse_full)/len(nmse_full), on_epoch=True, sync_dist=True)
        self.log(f"val/mean_ssim_full", sum(ssim_full)/len(nmse_full), on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_index):
        k_space = batch[0]
        ground_truth_image = batch[1]
        scaling_factor = batch[2]
        fully_sampled_k = k_space['fs_k_space']
        undersampled = k_space['undersampled']
        mask = k_space['mask']
        if (k_space['loss_mask'] * mask == 0).all(): # if disjoint masks, combine
            mask += k_space['loss_mask'] # combine to get original sampliing mask
        # pass inital data through model
        estimate_k = self.recon_model.pass_through_model(undersampled, mask, fully_sampled_k)
        estimate_k *= scaling_factor

        return super().test_step((estimate_k, ground_truth_image), batch_index)


    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        k_space = batch[0]
        ground_truth_image = batch[1]
        scaling_factor = batch[2]
        fully_sampled_k = k_space['fs_k_space']
        undersampled = k_space['undersampled']
        mask = k_space['mask']
        if (k_space['loss_mask'] * mask == 0).all(): # if disjoint masks, combine
            mask += k_space['loss_mask'] # combine to get original sampliing mask
        # pass inital data through model
        estimate_k = self.recon_model.pass_through_model(undersampled, mask, fully_sampled_k)
        estimate_k *= scaling_factor

        return super().on_test_batch_end(outputs, (estimate_k, ground_truth_image), batch_idx, dataloader_idx)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        #warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1) 
        #step_lr = StepLR(optimizer, step_size=50)
        return optimizer


    def _setup_image_space_loss(self, image_loss_function):
        if image_loss_function == 'ssim':
            image_loss = lambda targ, pred: 1 - self.ssim_func(targ, pred)
        elif image_loss_function == 'l1':
            l1_loss = torch.nn.L1Loss()
            image_loss = lambda targ, pred: l1_loss(targ, pred)
        elif image_loss_function == 'l1_grad':
            l1_loss = torch.nn.L1Loss()
            grad_x = lambda targ, pred: l1_loss(targ.diff(dim=-1), pred.diff(dim=-1))
            grad_y = lambda targ, pred: l1_loss(targ.diff(dim=-2), pred.diff(dim=-2))
            image_loss = lambda targ, pred: (
                    2*l1_loss(targ, pred) + 
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
            self.log(f'train/R_{contrast}', R_value[i])

    def log_k_space_set_ratios(self, input_mask, initial_mask):
        for i, contrast in enumerate(self.contrast_order):
            self.log(f'train/lambda-over-inverse_{contrast}', 
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
            self.image_scaling_lam_full
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
            self.image_scaling_lam_inv
            )
        # calculate inverse full loss image pathway
        if (inverse_k is not None) and (full_k is not None):
            loss_dict["image_loss_inverse_full"] = self.compute_image_loss(
            full_k, 
            inverse_k,
            undersampled_k,
            self.image_scaling_full_inv
            )
            
            
        return loss_dict
