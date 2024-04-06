from ml_recon.pl_modules.pl_loupe import LOUPE
from ml_recon.pl_modules.pl_undersampled import UndersampledDataset
from ml_recon.pl_modules.pl_varnet import pl_VarNet
from ml_recon.models import Unet
import torch

from functools import partial

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import os

def main():
    artifact_dir = WandbLogger.download_artifact(artifact="chiew-lab/MRI Reconstruction/model-8im9nqc1:v0")

    wandb_logger = WandbLogger(project='MRI Reconstruction', name='loupe_mask_output')


    
    checkpoint = torch.load(os.path.join(artifact_dir, 'model.ckpt'), map_location=torch.device('cpu'))
    print(checkpoint.keys())


    data_module = UndersampledDataset(**checkpoint['datamodule_hyper_parameters']) 
    data_module.setup(stage='train')
   

    backbone = partial(Unet, in_chan=8, out_chan=8, chans=18)
    recon_model = pl_VarNet(backbone, contrast_order=data_module.contrast_order, lr = 1e-3)

    model = LOUPE(recon_model=recon_model, **checkpoint['hyper_parameters'])
    model.load_state_dict(checkpoint['state_dict'])
    trainer = pl.Trainer(logger=wandb_logger)

    trainer.test(model, datamodule=data_module)


if __name__ == '__main__': 
    main()
