import os

from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning
from ml_recon.pl_modules.MRILoader import MRI_Loader
from ml_recon.pl_modules.pl_undersampled import UndersampledDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import Callback

def main():
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset_random_phase/'
    model_checkpoint = './artifacts/model-bd8hwfif:v0/model.ckpt'
        
    trainer = pl.Trainer()
    model = LearnedSSLLightning.load_from_checkpoint(model_checkpoint)
    datamodule_hparams = model.hparams.get("datamodule_hyper_parameters", {})

    # Instantiate the DataModule with the loaded hyperparameters
    datamodule = UndersampledDataset(**datamodule_hparams)

    trainer.test(model, datamodule=datamodule)


from ml_recon.utils import root_sum_of_squares, ifft_2d_img
import numpy as np
class SaveTestOutputs(Callback):
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_index):
        estimated_iamge = outputs
        k_space = batch['fs_k_space']
        images = root_sum_of_squares(ifft_2d_img(k_space, axes=(-1, -2)), coil_dim=2)
        images /= images.amax((-1, -2), keepdim=True)
        np.save(f'outputs/estimated_{batch_index}', np.squeeze(estimated_iamge))
        np.save(f'outputs/ground_truth{batch_index}', np.squeeze(images))

main()
