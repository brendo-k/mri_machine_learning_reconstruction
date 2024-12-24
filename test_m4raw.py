import torch
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
import os
from ml_recon.pl_modules.pl_varnet import pl_VarNet


def main():
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/M4raw_chunked/'
    logger = WandbLogger(project='M4Raw', name='More-Masked M4Raw ssl: t1')
    checkpoint = '/home/kadotab/python/ml/m4raw/o3q61m67/checkpoints/epoch=49-step=19200.ckpt'
    model = pl_VarNet.load_from_checkpoint(checkpoint)
    datamodule = UndersampledDataModule.load_from_checkpoint(checkpoint, data_dir=data_dir)

    trainer = pl.Trainer(logger=logger, accelerator='cuda')
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__': 
    main()
