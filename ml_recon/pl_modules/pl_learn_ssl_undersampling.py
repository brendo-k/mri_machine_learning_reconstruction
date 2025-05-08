from typing import Literal, Union
import dataclasses

import numpy as np
import os
import torch
from pytorch_lightning.loggers import WandbLogger


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from ml_recon.losses import L1L2Loss, SSIM_Loss, L1ImageGradLoss
from ml_recon.pl_modules.pl_ReconModel import plReconModel
from ml_recon.utils.evaluation_functions import nmse
from ml_recon.utils import k_to_img
from ml_recon.models.LearnPartitioning import LearnPartitioning, LearnPartitionConfig
from ml_recon.models.TriplePathway import TriplePathway, DualDomainConifg, VarnetConfig
from ml_recon.utils.evaluate_over_contrasts import evaluate_over_contrasts


class LearnedSSLLightning(plReconModel):
    def __init__(
        self,
        learn_partitioning_config: Union[LearnPartitionConfig, dict],
        varnet_config: Union[VarnetConfig, dict],
        dual_domain_config: Union[DualDomainConifg, dict],
        lr: float = 1e-3,
        lr_scheduler: bool = False,
        warmup_adam: bool = False,
        image_loss_scaling_lam_inv: float = 1e-4,
        image_loss_scaling_lam_full: float = 1e-4,
        image_loss_scaling_full_inv: float = 1e-4,
        lambda_scaling: float = 1,
        image_loss_function: str = "ssim",
        k_space_loss_function: Literal["l1l2", "l1", "l2"] = "l1l2",
        enable_learn_partitioning: bool = True,
        enable_warmup_training: bool = False,
        use_supervised_image_loss: bool = False,
        is_mask_testing: bool = True,
    ):
        """
        This function trains all MRI reconstruction models

        We train all models with this class. Supervised training is decided automatically based on the data given.

        We assume the dataloader has the keys:
        'undersampled': undersampled k-space from inital undersampling (Omega mask)
        'fs_k_space': fully sampled k-space
        'lambda_mask': One partition of k-space. If supervised, lambda_mask is the initial undersampling mask (Omega Mask)
        'loss_mask': Other partition of k-space. If supervised, loss_mask is all ones. 
        """

        # since we convert to dicts for uploading to wandb, we need to convert back to dataclasses
        # Needed when loading checkpoints
        if isinstance(learn_partitioning_config, dict):
            learn_partitioning_config = LearnPartitionConfig(
                **learn_partitioning_config
            )
        if isinstance(varnet_config, dict):
            varnet_config = VarnetConfig(**varnet_config)
        if isinstance(dual_domain_config, dict):
            dual_domain_config = DualDomainConifg(**dual_domain_config)

        super().__init__(
            contrast_order=varnet_config.contrast_order,
            is_mask_testing=is_mask_testing,
        )

        if enable_learn_partitioning:
            self.partition_model = LearnPartitioning(learn_partitioning_config)

        self.recon_model = TriplePathway(dual_domain_config, varnet_config)

        # convert to dicts because save hyperparameter method does not like dataclasses
        dual_domain_config = dataclasses.asdict(dual_domain_config)  # type: ignore
        varnet_config = dataclasses.asdict(varnet_config)  # type: ignore
        learn_partitioning_config = dataclasses.asdict(learn_partitioning_config)  # type: ignore

        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.warmup_adam = warmup_adam
        self.image_scaling_lam_inv = image_loss_scaling_lam_inv
        self.image_scaling_lam_full = image_loss_scaling_lam_full
        self.image_scaling_full_inv = image_loss_scaling_full_inv
        self.lambda_loss_scaling = lambda_scaling
        self.enable_warmup_training = enable_warmup_training
        self.enable_learn_partitioning = enable_learn_partitioning
        self.use_superviesd_image_loss = use_supervised_image_loss
        self.test_metrics = is_mask_testing

        
        # loss function init
        self._setup_image_space_loss(image_loss_function)
        self._setup_k_space_loss(k_space_loss_function)
        self.save_hyperparameters()

    def forward(self, k_space, mask, fs_k_space):
        estimate_k = self.recon_model.pass_through_model(
            k_space * mask, mask, fs_k_space
        )
        return estimate_k

    def training_step(self, batch, _):
        """
        Training loop function 

        This function loads data, partitions k-space if self-supervised then passes 
        data through triple pathways reconstruction network. The estimated outputs are
        then used to cacluate the loss function.

        Args:
            batch: dict, batch of data from above

        Returns:
            torch.Tensor, Returns final loss of this training batch

        Example:
            This is called internally by PyTorch Lightning
        """
        # get data
        fully_sampled = batch["fs_k_space"]
        undersampled_k = batch["undersampled"]

        # split data (loss mask is all ones if supervised)
        input_mask, loss_mask = self.partition_k_space(batch)

        # recon undersampled data
        estimates = self.recon_model.forward(
            undersampled_k,
            fully_sampled,
            input_mask,
            loss_mask,
            return_all=False,
        )

        # calculate loss
        loss_dict = self.calculate_loss(
            estimates, undersampled_k, fully_sampled, input_mask, loss_mask, "train"
        )
        loss: torch.Tensor = sum(loss for loss in loss_dict.values())  # type: ignore

        # log loss components
        for key, value in loss_dict.items():
            self.log_scalar(f"train/{key}", value)

        # log full loss
        self.log_scalar("train/loss", loss, on_step=True, prog_bar=True)

        # log R value (for partitioning)
        if self.enable_learn_partitioning:
            self.log_R_value()

        # log ratio of sets
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
            kwargs["on_step"] = False

        self.log(label, metric, on_epoch=True, **kwargs)

    # Plotting of different metrics during training
    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx): 
        """
        Hook to call when training batch is finished

        This function plots images and training  metrics for visualization on WandB

        Args:
            outputs: torch.Tensor, loss from training_step
            batch: dict, same from training_step
            batch_idx: int, Integer of batch during training loop

        Returns:
            None

        Example:
            Called internally by PyTorch Lightning        
        """
        if not isinstance(self.logger, WandbLogger):
            return

        #log first batch of every 1st epoch
        if batch_idx != 0 or self.current_epoch % 1 != 0:
           return

        wandb_logger = self.logger

        fully_sampled = batch["fs_k_space"]
        undersampled_k = batch["undersampled"]

        input_mask, loss_mask = self.partition_k_space(batch)
        estimates = self.recon_model.forward(
            undersampled_k, fully_sampled, input_mask, loss_mask, return_all=True
        )

        # plot images (first of the batch)
        image_scaling = k_to_img(fully_sampled).amax((-1, -2), keepdim=True)

        # convert to images and scale based on fully sampled (plotting needs [0-1])
        fully_sampled_images = self.k_to_img_scaled_clipped(
            fully_sampled, image_scaling
        )

        # plot estimated images from each path and their metrics compared to fully sampled
        for pathway, estimate in estimates.items():
            estimate_images = self.k_to_img_scaled_clipped(estimate, image_scaling)
            ssim_values = evaluate_over_contrasts(
                ssim, fully_sampled_images, estimate_images
            )
            nmse_values = evaluate_over_contrasts(
                nmse, fully_sampled_images, estimate_images
            )

            wandb_logger.log_image(
                f"train/estimate_{pathway}",
                self.split_along_contrasts(estimate_images[0]),
            )
            self.log(
                f"train_metrics/ssim_{pathway}", sum(ssim_values) / len(ssim_values)
            )
            self.log(
                f"train_metrics/nmse_{pathway}", sum(nmse_values) / len(nmse_values)
            )
            _, ssim_images = ssim(
                fully_sampled_images[0].unsqueeze(1),
                estimate_images[0].unsqueeze(1),
                data_range=(0, 1),
                return_full_image=True,
            )
            wandb_logger.log_image(
                f"train/ssim_image_{pathway}", self.split_along_contrasts(ssim_images)
            )

        # plot masks (first of the batch)
        initial_mask = (undersampled_k != 0)[0, :, 0, :, :]
        lambda_set_plot = input_mask[0, :, 0, :, :]
        loss_mask = loss_mask[0, :, 0, :, :]
        wandb_logger.log_image(
            "train_masks/lambda_set", self.split_along_contrasts(lambda_set_plot)
        )
        wandb_logger.log_image(
            "train_masks/loss_set", self.split_along_contrasts(loss_mask)
        )
        wandb_logger.log_image(
            "train_masks/initial_mask", self.split_along_contrasts(initial_mask)
        )

        # plot probability if learn partitioning
        if self.enable_learn_partitioning:
            probability = self.partition_model.get_probability_distribution()
            wandb_logger.log_image(
                "probability", self.split_along_contrasts(probability), self.global_step
            )

    def validation_step(self, batch, _):
        noisy_data = batch[0]
        undersampled_k = noisy_data["undersampled"]
        fully_sampled = noisy_data["fs_k_space"]

        input_mask, loss_mask = self.partition_k_space(noisy_data)
        estimates = self.recon_model.forward(
            undersampled_k,
            fully_sampled,
            input_mask,
            loss_mask,
            return_all=False,
        )

        # log loss
        loss_dict = self.calculate_loss(
            estimates, undersampled_k, fully_sampled, input_mask, loss_mask, "val"
        )
        loss = sum(loss for loss in loss_dict.values())

        self.log_scalar("val_losses/loss", loss, prog_bar=True, sync_dist=True)
        for key, value in loss_dict.items():
            self.log_scalar(f"val_losses/{key}", value)

        self.calculate_k_nmse(batch)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx > 4 or self.current_epoch % 10 != 0:
            plot_images = False
        else:
            plot_images = True

        estimate_k, fully_sampled_k = self.infer_k_space(batch)

        # get images
        estimate_image = k_to_img(estimate_k, coil_dim=2)
        fully_sampled_image = k_to_img(fully_sampled_k, coil_dim=2)
        ground_truth_image = batch[1]

        background_mask = self.get_image_background_mask(fully_sampled_image)

        scaling_factor = fully_sampled_image.amax((-1, -2), keepdim=True)

        estimate_image /= scaling_factor
        ground_truth_image /= scaling_factor
        fully_sampled_image /= scaling_factor

        estimate_image_masked = estimate_image * background_mask
        ground_truth_image_masked = ground_truth_image * background_mask
        fully_sampled_image_masked = fully_sampled_image * background_mask

        diff_masked = (estimate_image_masked - fully_sampled_image_masked).abs() * 10
        diff_unmasked = (estimate_image - fully_sampled_image).abs() * 10

        # log images
        if plot_images and isinstance(self.logger, WandbLogger):
            wandb_logger = self.logger
            images_to_plot = torch.cat((estimate_image_masked[0], estimate_image[0]))
            wandb_logger.log_image(
                f"val_images_recons/estimate_full_{batch_idx}",
                self.split_along_contrasts(images_to_plot.clip(0, 1)),
            )
            wandb_logger.log_image(
                f"val_images_diff/diff_masked{batch_idx}",
                self.split_along_contrasts(diff_masked.clip(0, 1)[0]),
            )
            wandb_logger.log_image(
                f"val_images_diff/diff_unmasked{batch_idx}",
                self.split_along_contrasts(diff_unmasked.clip(0, 1)[0]),
            )

            # plot ground truths at the first epoch
            if self.current_epoch == 0:
                wandb_logger.log_image(
                    f"val_images_target/fully_sampled_{batch_idx}",
                    self.split_along_contrasts(
                        fully_sampled_image_masked[0].clip(0, 1)
                    ),
                )

        # log image space metrics
        self.log_image_space_metrics(
            estimate_image, fully_sampled_image, ground_truth_image, label=""
        )
        self.log_image_space_metrics(
            estimate_image_masked,
            fully_sampled_image_masked,
            ground_truth_image_masked,
            label="masked",
        )

    def log_image_space_metrics(self, estimate_image, fully_sampled_image, ground_truth_image, label):
        ssim_full = evaluate_over_contrasts(ssim, fully_sampled_image, estimate_image)
        nmse_full = evaluate_over_contrasts(nmse, fully_sampled_image, estimate_image)
        for i, contrast in enumerate(self.contrast_order):
            self.log(
                f"val_ssim/{label}ssim_full_{contrast}",
                ssim_full[i],
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"val_nmse/{label}nmse_full_{contrast}",
                nmse_full[i],
                on_epoch=True,
                sync_dist=True,
            )
        self.log(
            f"val/{label}mean_nmse_full",
            sum(nmse_full) / len(nmse_full),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"val/{label}mean_ssim_full",
            sum(ssim_full) / len(nmse_full),
            on_epoch=True,
            sync_dist=True,
        )

        # log gt metrics
        ssim_full = evaluate_over_contrasts(ssim, ground_truth_image, estimate_image)
        nmse_full = evaluate_over_contrasts(nmse, ground_truth_image, estimate_image)
        for i, contrast in enumerate(self.contrast_order):
            self.log(
                f"GT_val_ssim/{label}ssim_full_{contrast}",
                ssim_full[i],
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"GT_val_nmse/{label}nmse_full_{contrast}",
                nmse_full[i],
                on_epoch=True,
                sync_dist=True,
            )
        self.log(
            f"val/{label}GT_mean_nmse_full",
            sum(nmse_full) / len(nmse_full),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"val/{label}GT_mean_ssim_full",
            sum(ssim_full) / len(nmse_full),
            on_epoch=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_index):
        estimate_k, fs_k = self.infer_k_space(batch)
        ground_truth_image = batch[1]  # averaged or denoised ground truth
        fully_sampled_image = k_to_img(
            fs_k, coil_dim=2
        )  # noisy, fully sampled ground truth

        fs_metrics = self.my_test_step(
            (estimate_k, fully_sampled_image), 
            batch_index, 
        )

        return {"fs_metrics": fs_metrics}

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        ground_truth_image = batch[1]
        estimate_k, fully_sampled = self.infer_k_space(batch)
        fully_sampled_image = k_to_img(
            fully_sampled, coil_dim=2
        )  # noisy, fully sampled ground truth

        self.my_test_batch_end(
            outputs,
            (estimate_k, ground_truth_image),
            batch_idx,
            dataloader_idx,
        )

    def on_train_epoch_start(self):
        # set epoch in train dataloader for updating new lambda masks
        if self.trainer.train_dataloader:
            self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch)
            self.trainer.val_dataloaders.dataset.undersampled_dataset.set_epoch(self.current_epoch)  # type: ignore
        return

    def infer_k_space(self, batch):
        k_space = batch[0]
        scaling_factor = batch[0]["scaling_factor"]
        fully_sampled_k = k_space["fs_k_space"].clone()
        undersampled = k_space["undersampled"].clone()
        mask = k_space["mask"].clone()

        # if self-supervised combine masks to pass all data.
        if k_space["is_self_supervised"].all():
            mask += k_space["loss_mask"]  # combine to get original sampling mask

        # pass inital data through model
        estimate_k = self.recon_model.pass_through_model(
            undersampled, mask, fully_sampled_k
        )

        # rescale based on scaling factor
        estimate_k *= scaling_factor
        fully_sampled_k *= scaling_factor
        return estimate_k, fully_sampled_k

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        schedulers = []
        if self.lr_scheduler:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=2000, T_mult=2, eta_min=1e-4
            )
            # step_lr = StepLR(optimizer, step_size=50, gamma=0.1)
            schedulers.append(
                {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            )
        return [optimizer], schedulers

    def calculate_k_nmse(self, batch):
        estimate_k, fully_sampled_k = self.infer_k_space(batch)
        mse_value = (fully_sampled_k - estimate_k).pow(2).abs().sum((-1, -2, -3))
        l2_norm = fully_sampled_k.pow(2).abs().sum((-1, -2, -3))
        self.log_scalar("val_nmse/k-space_nmse", (mse_value/l2_norm).mean())

    def _setup_image_space_loss(self, image_loss_function):
        if image_loss_function == "ssim":
            image_loss = SSIM_Loss(kernel_size=7, data_range=(0.0, 1.0))
        elif image_loss_function == "l1":
            image_loss = torch.nn.L1Loss()
        elif image_loss_function == "l1_grad":
            image_loss = L1ImageGradLoss(grad_scaling=0.5)
        else:
            raise ValueError(f"unsuported image loss function: {image_loss_function}")
        self.image_loss_func = image_loss

    def compute_image_loss(self, kspace1, kspace2, undersampled_k):
        scaling_factor = k_to_img(undersampled_k).amax((-1, -2), keepdim=True)
        img_1 = k_to_img(kspace1) / scaling_factor
        img_2 = k_to_img(kspace2) / scaling_factor

        b, c, h, w = img_1.shape
        img_1 = img_1.view(b * c, 1, h, w)
        img_2 = img_2.view(b * c, 1, h, w)
        image_loss = self.image_loss_func(img_1, img_2)

        return image_loss

    def calculate_k_loss(
        self, estimate, undersampled, loss_mask, loss_scaling, loss_name=""
    ):
        k_losses = {}
        for contrast, index in zip(self.contrast_order, range(estimate.shape[1])):
            k_loss = self.k_space_loss(
                torch.view_as_real(
                    undersampled[:, index, ...] * loss_mask[:, index, ...]
                ),
                torch.view_as_real(estimate[:, index, ...] * loss_mask[:, index, ...]),
            )
            # reduce mean by the loss mask and not by the number of voxels

            k_losses[f"k_loss_{loss_name}_{contrast}"] = k_loss * loss_scaling

        return k_losses

    def _setup_k_space_loss(self, k_space_loss_function):
        reduce = "mean"

        if k_space_loss_function == "l1l2":
            self.k_space_loss = L1L2Loss(norm_all_k=False)
        elif k_space_loss_function == "l1":
            self.k_space_loss = torch.nn.L1Loss(reduction=reduce)
        elif k_space_loss_function == "l2":
            self.k_space_loss = torch.nn.MSELoss(reduction=reduce)
        else:
            raise ValueError("No k-space loss!")

    def log_R_value(self):
        """
        Logs the R value for each contrast in the contrast order.

        The R value is obtained from the partition model and logged for each contrast.
        """
        R_value = self.partition_model.get_R()
        for i, contrast in enumerate(self.contrast_order):
            self.log(f"sampling_metrics/R_{contrast}", R_value[i])

    def log_k_space_set_ratios(self, input_mask, initial_mask):
        for i, contrast in enumerate(self.contrast_order):
            self.log(
                f"sampling_metrics/lambda-over-inverse_{contrast}",
                input_mask[:, i, 0, :, :].sum() / initial_mask[:, i, 0, :, :].sum(),
                on_epoch=True,
                on_step=False,
            )

    def calculate_inverse_k_loss(
        self, input_mask, loss_mask, inverse_k, undersampled_k
    ):
        _, lambda_k_wo_acs = TriplePathway.create_inverted_masks(
            input_mask,
            loss_mask,
            self.recon_model.dual_domain_config.pass_through_size,
            self.recon_model.dual_domain_config.pass_all_lines,
        )
        k_loss_inverse = self.calculate_k_loss(
            inverse_k,
            undersampled_k,
            lambda_k_wo_acs,
            (1 - self.lambda_loss_scaling),
            "inverse",
        )
        return k_loss_inverse

    def partition_k_space(self, batch):
        # compute either learned or heuristic partioning masks
        if self.enable_learn_partitioning:
            assert (batch["mask"] * batch["loss_mask"] == 0).all()
            initial_mask = batch["mask"] + batch["loss_mask"]
            input_mask, loss_mask = self.partition_model(initial_mask)
        else:
            input_mask, loss_mask = batch["mask"], batch["loss_mask"]

        return input_mask, loss_mask

    def split_along_contrasts(self, image):
        return np.split(image.cpu().detach().numpy(), image.shape[0], 0)

    def k_to_img_scaled_clipped(self, k_space, scaling_factor):
        return (k_to_img(k_space) / scaling_factor).clip(0, 1)

    def calculate_loss(
        self, estimates, undersampled_k, fully_sampled, input_mask, dc_mask, label
    ):
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
        lambda_k = estimates["lambda_path"]
        full_k = estimates["full_path"]
        inverse_k = estimates["inverse_path"]

        lam_full_scaling, lam_inv_scaling, inv_full_scaling = (
            self.get_image_space_scaling_factors()
        )

        loss_dict = {}
        if self.use_superviesd_image_loss:
            target_img = k_to_img(fully_sampled, coil_dim=2)
            lambda_img = k_to_img(lambda_k, coil_dim=2)
            ssim_val = ssim(
                target_img, lambda_img, data_range=(target_img.min().item(), target_img.max().item())
            )
            assert isinstance(ssim_val, torch.Tensor)
            loss_dict["image_loss"] = 1 - ssim_val

        # plot each loss individually
        k_losses = self.calculate_k_loss(
            lambda_k, fully_sampled, dc_mask, self.lambda_loss_scaling, "lambda"
        )

        for key, value in k_losses.items():
            self.log(f"{label}/{key}", value)

        loss_dict["k_loss"] = sum([values for values in k_losses.values()]) / len(k_losses)

        # calculate full lambda image loss pathway
        if full_k is not None:
            loss_dict["image_loss_full_lambda"] = self.compute_image_loss(
                full_k,
                lambda_k,
                undersampled_k,
            )
            self.log(
                f"{label}/unscaled_full_lambda", loss_dict["image_loss_full_lambda"]
            )
            loss_dict["image_loss_full_lambda"] *= lam_full_scaling

        # calculate inverse lambda image and k-space loss pathway
        if inverse_k is not None:
            # k space loss
            inverse_k_losses = self.calculate_inverse_k_loss(
                input_mask, dc_mask, inverse_k, undersampled_k
            )
            for key, value in inverse_k_losses.items():
                self.log(f"{label}/{key}", value)
            loss_dict["k_loss_inverse"] = sum([values for values in inverse_k_losses.values()]) / len(inverse_k_losses) # image space loss
            loss_dict["image_loss_inverse_lambda"] = self.compute_image_loss(
                lambda_k,
                inverse_k,
                undersampled_k,
            )
            self.log(
                "train/unscaled_inverse_lambda", loss_dict["image_loss_inverse_lambda"]
            )
            loss_dict["image_loss_inverse_lambda"] *= lam_inv_scaling
        # calculate inverse full loss image pathway
        if (inverse_k is not None) and (full_k is not None):
            loss_dict["image_loss_inverse_full"] = self.compute_image_loss(
                full_k,
                inverse_k,
                undersampled_k,
            )
            self.log(
                "train/unscaled_inverse_full", loss_dict["image_loss_inverse_full"]
            )
            loss_dict["image_loss_inverse_full"] *= inv_full_scaling

        return loss_dict

    def get_image_space_scaling_factors(self):
        warmup_epochs = int(os.getenv('WARMUP_EPOCHS')) if os.getenv('WARMUP_EPOCHS') else 10
        if self.enable_warmup_training and self.current_epoch < warmup_epochs:
            scaling_factor = self.current_epoch / warmup_epochs
        else:
            scaling_factor = 1

        return (
            scaling_factor * self.image_scaling_lam_full,
            scaling_factor * self.image_scaling_lam_inv,
            scaling_factor * self.image_scaling_full_inv,
        )
