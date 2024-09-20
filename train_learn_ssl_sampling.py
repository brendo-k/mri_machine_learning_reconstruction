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
            line_constrained=False, 
            R=args.R
            ) 

    data_module.setup('train')
    
    if args.line_constrained:
        prob_method = 'line_loupe'
    else:
        prob_method = 'loupe'

    model = LearnedSSLLightning(
            (len(args.contrasts), ny, nx), 
            learned_R=args.R_hat, 
            prob_method=prob_method,
            contrast_order=data_module.contrast_order,
            lr = args.lr,
            learn_R=args.learn_R,
            warm_start=args.warm_start,
            ssim_scaling=args.ssim_scaling,
            lambda_scaling=args.lambda_scaling,
            normalize_k_space_energy=args.k_space_regularizer,
            pass_all_data=args.pass_all_data,
            pass_inverse_data=args.pass_inverse_data,
            supervised=args.supervised
            )

    if args.checkpoint: 
        model = LearnedSSLLightning.load_from_checkpoint(os.path.join(args.checkpoint, 'model.ckpt'))

    if not args.learn_sampling:
        model.sampling_weights.requires_grad = False

    ## AUTOMATIC HYPERPARAMETER TUNING
    #tuner = Tuner(trainer)
    #tuner.scale_batch_size(model, mode='binsearch', datamodule=data_module)
    #tuner.lr_find(model, datamodule=data_module, min_lr=1e-4, max_lr=1e-1)

    #wandb_logger.experiment.config.update(model.hparams)

    print(model.hparams)
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__': 
    parser = ArgumentParser(description="Deep learning multi-contrast reconstruction")

    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--num_workers', type=int, default=0)
    training_group.add_argument('--max_epochs', type=int, default=50)
    training_group.add_argument('--batch_size', type=int, default=16)
    training_group.add_argument('--lr', type=float, default=1e-3)
    training_group.add_argument('--checkpoint', type=str)
    
    # dataset parameters
    dataset_group = parser.add_argument_group('Dataset Parameters')
    dataset_group.add_argument('--R', type=float, default=6.0)
    dataset_group.add_argument('--dataset', type=str, default='brats')
    dataset_group.add_argument('--contrasts', type=str, nargs='+', default=['t1', 't2', 't1ce', 'flair'])
    dataset_group.add_argument('--norm_method', type=str, default='k')
    dataset_group.add_argument('--nx', type=int, default=128)
    dataset_group.add_argument('--ny', type=int, default=128)
    dataset_group.add_argument('--limit_batches', type=float, default=1.0)
    dataset_group.add_argument('--line_constrained', action='store_true')

    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--R_hat', type=float, default=2.0)
    model_group.add_argument('--learn_R', action='store_true')
    model_group.add_argument('--warm_start', action='store_true')
    model_group.add_argument('--data_dir', type=str, default='/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset_random_phase/')
    model_group.add_argument('--R_seeding', type=float, nargs='+', default=[])
    model_group.add_argument('--R_freeze', type=bool, nargs='+', default=[])
    model_group.add_argument('--chans', type=int, default=32)
    model_group.add_argument('--ssim_scaling', type=float, default=0.0)
    model_group.add_argument('--lambda_scaling', type=float, default=0.0)
    model_group.add_argument('--k_space_regularizer', type=float, default=0.0)
    model_group.add_argument('--pass_inverse_data', action='store_true')
    model_group.add_argument('--pass_all_data', action='store_true')
    model_group.add_argument('--learn_sampling', action='store_true')
    model_group.add_argument('--supervised', action='store_true')

    logger_group = parser.add_argument_group('Logging Parameters')
    logger_group.add_argument('--project', type=str, default='MRI Reconstruction')
    logger_group.add_argument('--run_name', type=str)
    
    args = parser.parse_args()

    main(args)
