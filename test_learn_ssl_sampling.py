import os

from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
from pathlib import Path
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import Callback
from ml_recon.utils.image_slices import image_slices

def main():
    data_dir = '/home/brenden/Documents/data/simulated_subset_random_phase'
    model_checkpoint = '/home/brenden/Documents/code/python/mri_machine_learning_reconstruction-1/artifacts/model-9u43xz0p:v0/model.ckpt'
    wandb_logger = WandbLogger(project='MRI Reconstruction', name='ssl baseline')
    artifact = wandb_logger.use_artifact(artifact='chiew-lab/MRI Reconstruction/model-lyl2o7wz:v0')
    artifact_dir = artifact.download()
    trainer = pl.Trainer(callbacks=[], logger = wandb_logger)
    model = LearnedSSLLightning.load_from_checkpoint(Path(artifact_dir) / 'model.ckpt' )
    datamodule = UndersampledDataModule.load_from_checkpoint(Path(artifact_dir) / 'model.ckpt', data_dir=data_dir, batch_size=1)
    # Instantiate the DataModule with the loaded hyperparameters

    trainer.test(model, datamodule=datamodule)


from ml_recon.utils import root_sum_of_squares, ifft_2d_img
import numpy as np
import matplotlib.pyplot as plt
class SaveTestOutputs(Callback):
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_index):
        estimated_iamge = outputs
        k_space = batch['fs_k_space']
        images = root_sum_of_squares(ifft_2d_img(k_space, axes=(-1, -2)), coil_dim=2)
        images /= images.amax((-1, -2), keepdim=True)

        np.save(f'outputs/estimated_{batch_index}', np.squeeze(estimated_iamge.cpu().numpy()))
        np.save(f'outputs/ground_truth_{batch_index}', np.squeeze(images.cpu().numpy()))
        fig, ax = image_slices(np.squeeze(estimated_iamge.cpu().numpy(), axis=0), cmap='gray')
        fig.savefig(f'outputs/{batch_index}_estimated')
        plt.close()
        fig, ax = image_slices(np.squeeze(images.cpu().numpy(), axis=0), cmap='gray')
        fig.savefig(f'outputs/{batch_index}_ground_truth')
        plt.close()
        fig, ax = image_slices(np.squeeze((estimated_iamge - images).abs().cpu().numpy(), axis=0), cmap='gray', vmax=images.cpu().max()/4)
        fig.savefig(f'outputs/{batch_index}_diff')
        plt.close()


main()
