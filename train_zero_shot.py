from argparse import ArgumentParser
import os

from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning
from ml_recon.dataset.Zeroshot_datset import ZeroShotDataset
from ml_recon.pl_modules.pl_UndersampledDataModule import normalize_k_max, convert_dataclass_to_dict
from ml_recon.pl_modules.pl_varnet import pl_VarNet

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from torch.utils.data import DataLoader
import numpy as np

from torchvision.transforms import Compose
def main(args):
        
    wandb_logger = WandbLogger(project='Zero Shot', log_model=True, name=args.run_name)
    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         logger=wandb_logger, 
                         limit_train_batches=args.limit_batches,
                         limit_val_batches=args.limit_batches,
                         )

    transforms = Compose([normalize_k_max(), convert_dataclass_to_dict()])
    
    dataset_train = ZeroShotDataset(
        '/home/brenden/Documents/data/fastmri/train/file_brain_AXT1_201_6002688.h5',
        is_validation=False, 
        transforms=transforms,
        R=args.R,
        R_hat=args.R_hat
        )

    dataset_val = ZeroShotDataset(
        '/home/brenden/Documents/data/fastmri/train/file_brain_AXT1_201_6002688.h5',
        is_validation=True, 
        transforms=transforms,
        R=args.R,
        R_hat=args.R_hat
        )

    dataset_test = ZeroShotDataset(
        '/home/brenden/Documents/data/fastmri/train/file_brain_AXT1_201_6002688.h5',
        is_validation=False, 
        is_test=True,
        transforms=transforms,
        R=args.R,
        R_hat=args.R_hat
    )

    train_loader = DataLoader(
        dataset_train, batch_size=args.batch_size, pin_memory=True, shuffle=True
    )
    val_loader = DataLoader(
        dataset_val, batch_size=args.batch_size, pin_memory=True, shuffle=False
    )
    test_loader = DataLoader(
        dataset_test, batch_size=args.batch_size, pin_memory=True, shuffle=False
    )

    model = pl_VarNet(contrast_order=['t1'])

    if args.checkpoint: 
        model = LearnedSSLLightning.load_from_checkpoint(os.path.join(args.checkpoint, 'model.ckpt'))

    ## AUTOMATIC HYPERPARAMETER TUNING
    #tuner = Tuner(trainer)
    #tuner.scale_batch_size(model, mode='binsearch', datamodule=data_module)
    #tuner.lr_find(model, datamodule=data_module, min_lr=1e-4, max_lr=1e-1)

    #wandb_logger.experiment.config.update(model.hparams)

    print(model.hparams)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == '__main__': 
    parser = ArgumentParser(description="Deep learning multi-contrast reconstruction")

    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--num_workers', type=int, default=0)
    training_group.add_argument('--max_epochs', type=int, default=50)
    training_group.add_argument('--batch_size', type=int, default=1)
    training_group.add_argument('--lr', type=float, default=1)
    training_group.add_argument('--checkpoint', type=str)
    
    # dataset parameters
    dataset_group = parser.add_argument_group('Dataset Parameters')
    dataset_group.add_argument('--R', type=float, default=4.0)
    dataset_group.add_argument('--dataset', type=str, default='brats')
    dataset_group.add_argument('--contrasts', type=str, nargs='+', default=['t1', 't2', 't1ce', 'flair'])
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
    model_group.add_argument('--ssim_scaling_full', type=float, default=0.0)
    model_group.add_argument('--ssim_scaling_set', type=float, default=0.0)
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
