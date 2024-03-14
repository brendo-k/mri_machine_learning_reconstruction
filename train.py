from argparse import ArgumentParser
import torch

from ml_recon.pl_modules.pl_varnet import pl_VarNet
from ml_recon.models.unet import Unet
from ml_recon.pl_modules.pl_loupe import LOUPE
from ml_recon.pl_modules.pl_supervised import SupervisedDataset
from ml_recon.pl_modules.pl_self_supervsied import SelfSupervisedDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from functools import partial


def main(args):
    torch.autograd.set_detect_anomaly(True)
    tb_logger = TensorBoardLogger('tb_logs', default_hp_metric=False)
    csv_logger = CSVLogger('csv_logs')
    wandb_logger = WandbLogger(project='MRI Reconstruction', log_model=True)
    trainer = pl.Trainer(max_epochs=args.max_epochs, logger=[tb_logger, csv_logger, wandb_logger], limit_train_batches=args.limit_train_batches)


    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset_random_phase/'
    nx = args.nx
    ny = args.ny
    
    if args.supervised:
        data_module = SupervisedDataset(
                'brats', 
                data_dir, 
                batch_size=args.batch_size, 
                resolution=(ny, nx),
                num_workers=args.num_workers,
                norm_method=args.norm_method,
                R=args.R,
                line_constrained=args.line_constrained,
                segregated=args.segregated
                ) 
    else:
        data_module = SelfSupervisedDataset(
                'brats', 
                data_dir, 
                batch_size=args.batch_size, 
                resolution=(ny, nx),
                num_workers=args.num_workers,
                norm_method=args.norm_method,
                R=args.R,
                line_constrained=args.line_constrained,
                segregated=args.segregated
                )


    data_module.setup('train')
    
    backbone = partial(Unet, in_chan=8, out_chan=8, chans=18)
    model = pl_VarNet(backbone, contrast_order=data_module.contrast_order, lr = args.lr)
    if args.learn_sampling:
        if args.line_constrained:
            prob_method = 'line_loupe'
        else:
            prob_method = 'loupe'

        model = LOUPE(
                model, 
                (4, ny, nx), 
                learned_R=args.R, 
                prob_method=prob_method,
                contrast_order=data_module.contrast_order,
                lr = args.lr,
                mask_method=args.mask_method, 
                lambda_param=args.lambda_param
                )

    # AUTOMATIC HYPERPARAMETER TUNING
    #tuner = Tuner(trainer)
    #tuner.scale_batch_size(model, mode='binsearch', datamodule=data_module)
    #tuner.lr_find(model, datamodule=data_module, min_lr=1e-4, max_lr=1e-1)

    wandb_logger.experiment.config.update(model.hparams)

    print(model.hparams)
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__': 
    parser = ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--learn_sampling', action='store_true')
    parser.add_argument('--line_constrained', action='store_true')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mask_method', type=str, default='all')
    parser.add_argument('--R', type=int, default=6)
    parser.add_argument('--lambda_param', type=float, default=0.)
    parser.add_argument('--limit_train_batches', type=float, default=1.0)
    parser.add_argument('--segregated', action='store_true')
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--ny', type=int, default=128)
    parser.add_argument('--norm_method', type=str, default='k')
    parser.add_argument('--supervised', action='store_true')
    
    args = parser.parse_args()

    main(args)
