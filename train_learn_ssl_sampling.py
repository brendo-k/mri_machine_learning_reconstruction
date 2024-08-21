from argparse import ArgumentParser
import os

from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning
from ml_recon.pl_modules.MRILoader import MRI_Loader
from ml_recon.pl_modules.pl_undersampled import UndersampledDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.cli import LightningCLI


def main(args):
        
    wandb_logger = WandbLogger(project='MRI Reconstruction', log_model=True, name=args.run_name)
    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         logger=wandb_logger, 
                         limit_train_batches=args.limit_batches,
                         limit_val_batches=args.limit_batches,
                         )


    data_dir = args.data_dir
    nx = args.nx
    ny = args.ny
    
    data_module = UndersampledDataset(
            args.dataset, 
            data_dir, 
            batch_size=args.batch_size, 
            resolution=(ny, nx),
            num_workers=args.num_workers,
            norm_method=args.norm_method,
            contrasts=args.contrasts,
            line_constrained=False
            ) 

    data_module.setup('train')
    
    if args.line_constrained:
        prob_method = 'line_loupe'
    else:
        prob_method = 'loupe'

    model = LearnedSSLLightning(
            (len(args.contrasts), ny, nx), 
            learned_R=args.R, 
            prob_method=prob_method,
            contrast_order=data_module.contrast_order,
            lr = args.lr,
            learn_R=args.learn_R,
            warm_start=args.warm_start,
            self_supervised=args.self_supervised,
            ssim_scaling=args.ssim_scaling,
            lambda_scaling=args.lambda_scaling,
            normalize_k_space_energy=args.k_space_regularizer
            )

    if args.checkpoint: 
        model = LearnedSSLLightning.load_from_checkpoint(os.path.join(args.checkpoint, 'model.ckpt'))

    ## AUTOMATIC HYPERPARAMETER TUNING
    #tuner = Tuner(trainer)
    #tuner.scale_batch_size(model, mode='binsearch', datamodule=data_module)
    #tuner.lr_find(model, datamodule=data_module, min_lr=1e-4, max_lr=1e-1)

    #wandb_logger.experiment.config.update(model.hparams)

    print(model.hparams)
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__': 
    parser = ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--learn_sampling', action='store_true')
    parser.add_argument('--line_constrained', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--R', type=float, default=6.0)
    parser.add_argument('--R_hat', type=float, default=2.0)
    parser.add_argument('--lambda_param', type=float, default=0.)
    parser.add_argument('--limit_batches', type=float, default=1.0)
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--ny', type=int, default=128)
    parser.add_argument('--norm_method', type=str, default='k')
    parser.add_argument('--self_supervised', action='store_true')
    parser.add_argument('--fd_param', type=float, default=0)
    parser.add_argument('--learn_R', action='store_true')
    parser.add_argument('--warm_start', action='store_true')
    parser.add_argument('--data_dir', type=str, default='/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset_random_phase/')
    parser.add_argument('--contrasts', type=str, nargs='+', default=['t1', 't2', 't1ce', 'flair'])
    parser.add_argument('--R_seeding', type=float, nargs='+', default=[])
    parser.add_argument('--R_freeze', type=bool, nargs='+', default=[])
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--project', type=str, default='MRI Reconstruction')
    parser.add_argument('--dataset', type=str, default='brats')
    parser.add_argument('--chans', type=int, default=32)
    parser.add_argument('--ssim_scaling', type=float, default=0.0)
    parser.add_argument('--lambda_scaling', type=float, default=0.0)
    parser.add_argument('--k_space_regularizer', type=float, default=0.0)

    
    args = parser.parse_args()

    main(args)
