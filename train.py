from argparse import ArgumentParser

from ml_recon.pl_modules.pl_varnet import pl_VarNet
from ml_recon.models.unet import Unet
from ml_recon.pl_modules.pl_loupe import LOUPE
from ml_recon.pl_modules.pl_undersampled import UndersampledDataset
from ml_recon.models import Unet

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from functools import partial

def main(args):
    wandb_logger = WandbLogger(project='MRI Reconstruction', log_model=True)
    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         logger=wandb_logger, 
                         limit_train_batches=args.limit_batches,
                         limit_val_batches=args.limit_batches,
                         devices="auto", 
                         strategy="auto"
                         )


    data_dir = args.data_dir
    nx = args.nx
    ny = args.ny
    
    data_module = UndersampledDataset(
            'brats', 
            data_dir, 
            batch_size=args.batch_size, 
            resolution=(ny, nx),
            num_workers=args.num_workers,
            norm_method=args.norm_method,
            R=args.R,
            R_hat=args.R_hat,
            line_constrained=args.line_constrained,
            self_supervsied=args.self_supervised,
            contrasts=args.contrasts
            ) 

    data_module.setup('train')
    
    model = pl_VarNet(contrast_order=data_module.contrast_order, lr = args.lr, num_cascades=args.cascades, chans=args.chans)

    ## AUTOMATIC HYPERPARAMETER TUNING
    #tuner = Tuner(trainer)
    #tuner.scale_batch_size(model, mode='binsearch', datamodule=data_module)
    #tuner.lr_find(model, datamodule=data_module, min_lr=1e-4, max_lr=1e-1)

    #wandb_logger.experiment.config.update(model.hparams)

    print(data_module.hparams)
    print(model.hparams)
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__': 
    parser = ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--line_constrained', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--R', type=float, default=6.0)
    parser.add_argument('--R_hat', type=float, default=2.0)
    parser.add_argument('--limit_batches', type=float, default=1.0)
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--ny', type=int, default=128)
    parser.add_argument('--norm_method', type=str, default='k')
    parser.add_argument('--self_supervised', action='store_true')
    parser.add_argument('--data_dir', type=str, default='/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset_random_phase/')
    parser.add_argument('--contrasts', type=str, nargs='+', default=['t1', 't2', 't1ce', 'flair'])
    parser.add_argument('--chans', type=int, default=32)
    parser.add_argument('--cascades', type=int, default=6)
    
    args = parser.parse_args()

    main(args)
