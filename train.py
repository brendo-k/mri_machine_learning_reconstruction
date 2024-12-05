from argparse import ArgumentParser
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from ml_recon.pl_modules.pl_varnet import pl_VarNet, VarnetConfig
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
from ml_recon.utils import replace_args_from_config



"""
Training code for multicontrast dataset

This code takes a multi--contrast datset and trains a VarNet architceture

Examples:
    # Basic training
    train.py --num_workers 3 --max_epochs 50 --contrasts t1 t2 flair
"""
def main(args):
    pl.seed_everything(8)
    torch.set_float32_matmul_precision('medium')

    wandb_logger = WandbLogger(project=args.project, name=args.run_name, log_model=True, save_dir='/home/kadotab/scratch/')
    #unique_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    #file_name = 'pl_varnet-' + unique_id
    #checkpoint_callback = ModelCheckpoint(
    #    dirpath='checkpoints/',  # Directory to save the checkpoints
    #    filename=file_name + '-{epoch:02d}-{val_loss:.2f}',  # Filename pattern
    #    monitor="val/val_loss", 
    #    mode="min", 
    #    save_last=True, 
    #    )


    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         logger=wandb_logger, 
                         limit_train_batches=args.limit_batches,
                         limit_val_batches=args.limit_batches,
                         callbacks=[],
                         )

    if args.checkpoint: 
        print("Loading Checkpoint!")
        model = pl_VarNet.load_from_checkpoint(args.checkpoint)
        data_module = UndersampledDataModule.load_from_checkpoint(args.checkpoint, data_dir=args.data_dir)
        data_module.setup('train')

    else:
        data_module = UndersampledDataModule(
                dataset_name=args.dataset, 
                data_dir=args.data_dir, 
                batch_size=args.batch_size, 
                resolution=(args.ny, args.nx),
                num_workers=args.num_workers,
                norm_method=args.norm_method,
                R=args.R,
                R_hat=args.R_hat,
                line_constrained=args.line_constrained,
                supervised_dataset=args.supervised,
                pi_sampling=args.pi_sampling, 
                contrasts=args.contrasts, 
                ssdu_partioning=args.ssdu_partioning, 
                ) 
        # this needs to be done to load contrast ordering for model
        data_module.setup('train')

        model_config = VarnetConfig(
            model_name=args.model, 
            contrast_order=data_module.contrast_order,
            lr = args.lr, 
            cascades=args.cascades, 
            channels=args.chans,
            norm_all_k=args.norm_all_k,
            image_loss_function=args.image_space_loss,
            image_loss_scaling=args.image_loss_scaling,
            k_loss_function=args.k_loss, 

        )
        model = pl_VarNet(config=model_config)




    print(data_module.hparams)
    print(model.hparams)
    trainer.fit(model=model, datamodule=data_module, ckpt_path=args.checkpoint)
    trainer.test(model, datamodule=data_module)
    
    

if __name__ == '__main__': 
    parser = ArgumentParser()

    # training arguments
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument("--config", type=str, help="Path to the YAML configuration file.")

    # dataset arguments
    parser.add_argument('--limit_batches', type=float, default=1.0)
    parser.add_argument('--R', type=float, default=4.0)
    parser.add_argument('--R_hat', type=float, default=2.0)
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--ny', type=int, default=128)
    parser.add_argument('--norm_method', type=str, default='k')
    parser.add_argument('--supervised', action='store_true')
    parser.add_argument('--data_dir', type=str, default='/home/brenden/Documents/data/simulated_subset_random_phase')
    parser.add_argument('--dataset', type=str, default='brats')
    parser.add_argument('--contrasts', type=str, nargs='+', default=['t1', 't2', 't1ce', 'flair'])
    parser.add_argument('--ssdu_partioning', action='store_true')
    parser.add_argument('--line_constrained', action='store_true')

    # model arguments
    parser.add_argument('--chans', type=int, default=32)
    parser.add_argument('--cascades', type=int, default=6)
    parser.add_argument('--pi_sampling', action='store_true')
    parser.add_argument('--norm_all_k', action='store_true')
    parser.add_argument('--image_space_loss', type=str, default='')
    parser.add_argument('--k_loss', type=str, default='norml1l2')
    parser.add_argument('--image_loss_scaling', type=float, default=0)

    # loggin arguments
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--project', type=str, default='MRI Reconstruction')

    args = parser.parse_args()

    args = replace_args_from_config(args.config, args)


    main(args)
