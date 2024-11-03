import torch
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
import os
from ml_recon.pl_modules.pl_varnet import pl_VarNet
from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning


def main():
    data_dir = '/home/kadotab/scratch/simulated_brats_1e-3_10/'
    logger = WandbLogger(project='SSL Characterization', name='mc ssl, 1e-3')
    artifact = logger.use_artifact('chiew-lab/MRI Reconstruction/model-pe7ert72:v1')
    artifact_dir = artifact.download()
    model = pl_VarNet.load_from_checkpoint(os.path.join(artifact_dir, 'model.ckpt'))
    datamodule = UndersampledDataModule.load_from_checkpoint(os.path.join(artifact_dir, 'model.ckpt'), data_dir=data_dir, batch_size=10)

    trainer = pl.Trainer(logger=logger, accelerator='cuda')
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__': 
    main()
