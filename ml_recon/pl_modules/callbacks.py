import torch
import numpy as np
from torchvision.utils import make_grid
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger

from ml_recon.utils import k_to_img, root_sum_of_squares, ifft_2d_img
from ml_recon.models.TriplePathway import TriplePathway
from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning
from ml_recon.utils.mask_background import get_image_background_mask


class TrainingPlottingCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not isinstance(trainer.logger, WandbLogger):
            return
        
        if not isinstance(pl_module, LearnedSSLLightning):
            return

        # log first batch of every 1st epoch
        if batch_idx != 0 or trainer.current_epoch % 1 != 0:
            return

        wandb_logger = trainer.logger
        fully_sampled = batch["fs_k_space"]
        undersampled_k = batch["undersampled"]

        input_mask, loss_mask = pl_module.partition_k_space(batch)
        estimates = pl_module.model.forward(
            undersampled_k, fully_sampled, input_mask, loss_mask, return_all=True
        )

        # plot images (first of the batch)
        image_scaling = k_to_img(fully_sampled).amax((-1, -2), keepdim=True)

        # plot estimated images from each path
        for pathway, estimate in estimates.items():
            estimate_images = pl_module.k_to_img_scaled_clipped(estimate, image_scaling)
            wandb_logger.log_image(f"train/estimate_{pathway}", [prep_for_plotting(estimate_images)])

        # plot masks (first of the batch)
        initial_mask = (undersampled_k != 0)[0, :, 0, :, :]
        lambda_set_plot = input_mask[0, :, 0, :, :]
        loss_mask_plot = loss_mask[0, :, 0, :, :]
        loss_mask_wo_acs, lambda_k_wo_acs = TriplePathway.create_inverted_masks(
            input_mask,
            loss_mask,
            pl_module.model.dual_domain_config.pass_through_size,
            pl_module.model.dual_domain_config.pass_all_lines,
        )
        wandb_logger.log_image(
            "train_masks/lambda_set", pl_module.split_along_contrasts(lambda_set_plot)
        )
        wandb_logger.log_image(
            "train_masks/loss_set", pl_module.split_along_contrasts(loss_mask_plot)
        )
        wandb_logger.log_image(
            "train_masks/lambda_set_inverse", pl_module.split_along_contrasts(lambda_k_wo_acs[0, :, 0, :, :])
        )
        wandb_logger.log_image(
            "train_masks/loss_set_inverse", pl_module.split_along_contrasts(loss_mask_wo_acs[0, :, 0, :, :])
        )
        wandb_logger.log_image(
            "train_masks/initial_mask", pl_module.split_along_contrasts(initial_mask)
        )

        # plot probability if learn partitioning
        if pl_module.enable_learn_partitioning and pl_module.partition_model:
            probability = pl_module.partition_model.get_probability_distribution()
            wandb_logger.log_image("probability", pl_module.split_along_contrasts(probability))


class ValidationPlottingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.plotting_indecies = [8, 16, 24, 32] # indecies of samples in the batch to plot (plots first 4 samples)
    def on_validation_epoch_end(self, trainer, pl_module):
        if not isinstance(trainer.logger, WandbLogger):
            return
        if not isinstance(pl_module, LearnedSSLLightning):
            return
        
        # Collect exactly 4 samples from the validation dataloader, irrespective of batch size
        batch = self.get_4batch_for_plotting(trainer, set='val')

        # infer
        estimate_k, fully_sampled_k, _ = pl_module.infer_k_space(batch)

        # get images from estimates
        fully_sampled_image_masked, estimated_image_masked, diff_masked = build_imgs_and_diff(fully_sampled_k, estimate_k)

        # log images
        wandb_logger = trainer.logger
        wandb_logger.log_image(f"val_images/estimate", [prep_for_plotting(estimated_image_masked)])
        wandb_logger.log_image(f"val_images/diff_masked", [prep_for_plotting(diff_masked)])

        # plot ground truths at the first epoch
        if trainer.current_epoch == 0:
            wandb_logger.log_image(f"val_images/fully_sampled", [prep_for_plotting(fully_sampled_image_masked)])



    # grabs a batch of size 4 for plotting, regardless of the batch size of the dataloader
    def get_4batch_for_plotting(self, trainer, set='val'):
        if set == 'val':
            dataloader = trainer.datamodule.val_dataloader().dataset
        elif set == 'test':
            dataloader = trainer.datamodule.test_dataloader().dataset
        else:
            raise NotImplementedError(f'{set} is not implemented, choose from val or test')
        
        data = []
        for i in self.plotting_indecies:
            data.append(dataloader[i])

        # stack data into a batch
        batch = {}
        for key in data[0].keys():
            batch[key] = torch.stack([d[key].to(trainer.strategy.root_device) for d in data], dim=0)

        return batch


class TestPlottingCallback(Callback):
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not isinstance(trainer.logger, WandbLogger):
            return
        if not isinstance(pl_module, LearnedSSLLightning):
            return
        
        logger = trainer.logger

        estimate_k, fully_sampled, mask = pl_module.infer_k_space(batch)
        ground_truth_image_masked, estimated_image_masked, difference_image_masked = build_imgs_and_diff(fully_sampled, estimate_k)
        
        logger.log_image(f'test/masked_recon', [prep_for_plotting(estimated_image_masked)])
        logger.log_image(f'test/masked_target', [prep_for_plotting(ground_truth_image_masked)])
        logger.log_image(f'test/masked_diff', [prep_for_plotting(difference_image_masked)])

        trainer.logger.log_image(f'test/undersampling_mask', [prep_for_plotting(mask[:, :, 0])])


def prep_for_plotting(data):
    data = data.clamp(0, 1)
    b, contrast, h, w = data.shape
    data = data.reshape(-1, 1, h, w)
    data = make_grid(data, nrow=contrast, pad_value=1)
    return data

# converts k-space to images and masks. Then get difference.
def build_imgs_and_diff(fully_sampled_k, estimate_k):
     # get images
    estimate_image = k_to_img(estimate_k, coil_dim=2)
    fully_sampled_image = k_to_img(fully_sampled_k, coil_dim=2)

    # scale image to 0, 1
    scaling_factor = fully_sampled_image.amax((-1, -2), keepdim=True)
    estimate_image /= scaling_factor
    fully_sampled_image /= scaling_factor

    # apply background mask to images and difference
    background_mask = get_image_background_mask(fully_sampled_image)
    estimate_image_masked = estimate_image * background_mask
    fully_sampled_image_masked = fully_sampled_image * background_mask

    # get difference image and scale for visibility
    diff_masked = (estimate_image_masked - fully_sampled_image_masked).abs() * 10

    return fully_sampled_image_masked, estimate_image_masked, diff_masked