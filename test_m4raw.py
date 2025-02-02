import torch
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
import os
from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning
import wandb


def main():
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/M4raw_chunked/'
    test_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/M4raw_chunked//'
    checkpoint = '/home/kadotab/python/ml/checkpoints/last.ckpt'
    model = LearnedSSLLightning.load_from_checkpoint(checkpoint)
    datamodule = UndersampledDataModule.load_from_checkpoint(checkpoint, data_dir=data_dir, test_dir=test_dir)
    
    config = dict(model.hparams)
    config.update(dict(datamodule.hparams))

    wandb.init(project='Figure 3 M4Raw', name='0', config=config)
    logger = WandbLogger(project='Figure 3 M4Raw', name='')

    trainer = pl.Trainer(logger=logger, accelerator='cuda')
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__': 
    main()
