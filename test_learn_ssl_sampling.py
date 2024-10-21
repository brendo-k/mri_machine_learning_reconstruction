import os

from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import sys

def main():
    artifact_path = sys.argv[1]
    wandb_logger = WandbLogger(project='MRI Reconstruction', name='ssl baseline')
    artifact = wandb_logger.use_artifact(artifact=artifact_path)
    artifact_dir = artifact.download()
    trainer = pl.Trainer(callbacks=[], logger = wandb_logger)
    model = LearnedSSLLightning.load_from_checkpoint(Path(artifact_dir) / 'model.ckpt' )
    datamodule = UndersampledDataModule.load_from_checkpoint(Path(artifact_dir) / 'model.ckpt', batch_size=1)
    # Instantiate the DataModule with the loaded hyperparameters

    trainer.test(model, datamodule=datamodule)


main()
