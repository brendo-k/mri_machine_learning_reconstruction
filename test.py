import torch
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
import os
from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning
import argparse


def main(args):
    pl.seed_everything(8)
    data_dir = args.data_dir
    test_dir = args.test_dir
    checkpoint_path = args.checkpoint
    logger = WandbLogger(project='MRI Reconstruction')
    if args.wandb_artifact:
        artifact = logger.use_artifact(args.wandb_artifact)
        artifact_dir = artifact.download()
        checkpoint_path = os.path.join(artifact_dir, 'model.ckpt')
    
    model = LearnedSSLLightning.load_from_checkpoint(checkpoint_path)
    datamodule = UndersampledDataModule.load_from_checkpoint(checkpoint_path, test_dir=test_dir, batch_size=10, data_dir=data_dir)

    trainer = pl.Trainer(logger=logger, accelerator='cuda')
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--wandb_artifact', type=str)
    
    args = parser.parse_args()    
    main(args)
