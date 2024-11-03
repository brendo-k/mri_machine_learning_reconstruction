from argparse import ArgumentParser

from ml_recon.pl_modules.pl_varnet import pl_VarNet
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner


"""
Training code for multicontrast dataset

This code takes a multi--contrast datset and trains a VarNet architceture

Examples:
    # Basic training
    train.py --num_workers 3 --max_epochs 50 --contrasts t1 t2 flair
"""
def main(args):
    wandb_logger = WandbLogger(project=args.project, name=args.run_name, log_model=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',  # Directory to save the checkpoints
        filename='mri-reconstruction-{epoch:02d}-{val_loss:.2f}',  # Filename pattern
        monitor="val/val_loss", 
        mode="min", 
        save_last=True, 
        save_weights_only=True
        )
    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         logger=wandb_logger, 
                         limit_train_batches=args.limit_batches,
                         limit_val_batches=args.limit_batches,
                         callbacks=[checkpoint_callback],
                         devices="auto", 
                         strategy="auto"
                         )


    data_dir = args.data_dir
    nx = args.nx
    ny = args.ny
    
    data_module = UndersampledDataModule(
            args.dataset_name, 
            data_dir, 
            batch_size=args.batch_size, 
            resolution=(ny, nx),
            num_workers=args.num_workers,
            norm_method=args.norm_method,
            R=args.R,
            R_hat=args.R_hat,
            line_constrained=args.line_constrained,
            self_supervsied=args.self_supervised,
            is_variable_density=args.pi_sampling, 
            contrasts=args.contrasts
            ) 

    data_module.setup('train')
    
    model = pl_VarNet(
            model_name=args.model, 
            contrast_order=data_module.contrast_order,
            lr = args.lr, 
            cascades=args.cascades, 
            chans=args.chans
            )



    #wandb_logger.experiment.config.update(model.hparams)
   
    #wandb_logger.experiment.config.update(model.hparams)

    print(data_module.hparams)
    print(model.hparams)
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__': 
    parser = ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--line_constrained', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--R', type=float, default=4.0)
    parser.add_argument('--R_hat', type=float, default=2.0)
    parser.add_argument('--limit_batches', type=float, default=1.0)
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--ny', type=int, default=128)
    parser.add_argument('--norm_method', type=str, default='k')
    parser.add_argument('--self_supervised', action='store_true')
    parser.add_argument('--data_dir', type=str, default='/home/brenden/Documents/data/simulated_subset_random_phase')
    parser.add_argument('--contrasts', type=str, nargs='+', default=['t1', 't2', 't1ce', 'flair'])
    parser.add_argument('--chans', type=int, default=32)
    parser.add_argument('--cascades', type=int, default=6)
    parser.add_argument('--dataset_name', type=str, default='brats')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--project', type=str, default='MRI Reconstruction')
    parser.add_argument('--pi_sampling', action='store_false')
    
    args = parser.parse_args()

    main(args)
