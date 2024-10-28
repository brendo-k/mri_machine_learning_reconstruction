import torch
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
import os
from ml_recon.pl_modules.pl_varnet import pl_VarNet


def main():
    data_dir = '/home/brenden/Documents/data/m4raw'
    logger = WandbLogger(project='SSL Characterization', name='t1 Supervised')
    artifact = logger.use_artifact('chiew-lab/SSL Characterization/model-2x40fhbe:v0')
    artifact_dir = artifact.download()
    model = pl_VarNet.load_from_checkpoint(os.path.join(artifact_dir, 'model.ckpt'))
    datamodule = UndersampledDataModule.load_from_checkpoint(os.path.join(artifact_dir, 'model.ckpt'), data_dir=data_dir)

    trainer = pl.Trainer(logger=logger, accelerator='cuda')
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__': 
    main()
