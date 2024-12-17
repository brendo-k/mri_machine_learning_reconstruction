from argparse import ArgumentParser
from datetime import datetime

import torch
from torch.profiler import ProfilerActivity, schedule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler

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

    wandb_logger = WandbLogger(project=args.project, name=args.run_name, log_model=True, save_dir='/home/kadotab/scratch')
    #unique_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    #file_name = 'pl_varnet-' + unique_id
    #checkpoint_callback = ModelCheckpoint(
    #    dirpath='checkpoints/',  # Directory to save the checkpoints
    #    filename=file_name + '-{epoch:02d}-{val_loss:.2f}',  # Filename pattern
    #    monitor="val/val_loss", 
    #    mode="min", 
    #    save_last=True, 
    #    )
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] 
    prof_scheduler = schedule(
        warmup=2,
        active=5, 
        skip_first=5,
        wait=0,
    )
    pytorch_profiler = PyTorchProfiler(
        activities=activities,
        schedule=prof_scheduler,
        export_to_chrome=True,
        dirpath='.',
        filename='prof'
    )

    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         logger=wandb_logger, 
                         limit_train_batches=args.limit_batches,
                         limit_val_batches=args.limit_batches,
                         callbacks=[],
                         #precision='bf16-mixed'
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
                contrasts=args.contrasts, 
                ) 
        # this needs to be done to load contrast ordering for model
        data_module.setup('train')

        varnet_config = VarnetConfig(
            contrast_order=data_module.contrast_order,
            cascades=args.cascades, 
            channels=args.chans,
            split_contrast_by_phase=args.split_contrast_by_phase,
            sensetivity_estimation=args.sense_method,
            model=args.model

        )

        model = pl_VarNet(
            config=varnet_config, 
            lr = args.lr, 
            norm_all_k=args.norm_all_k,
            image_loss_function=args.image_space_loss,
            image_loss_scaling=args.image_loss_scaling,
            k_loss_function=args.k_loss, 
            is_supervised=args.supervised
            )
            

    print(data_module.hparams)
    print(model.hparams)
    #model = torch.compile(model)
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
    parser.add_argument('--nx', type=int, default=256)
    parser.add_argument('--ny', type=int, default=256)
    parser.add_argument('--norm_method', type=str, default='k')
    parser.add_argument('--supervised', action='store_true')
    parser.add_argument('--data_dir', type=str, default='/home/brenden/Documents/data/simulated_subset_random_phase')
    parser.add_argument('--dataset', type=str, default='brats')
    parser.add_argument('--contrasts', type=str, nargs='+', default=['t1', 't2', 't1ce', 'flair'])
    parser.add_argument('--sampling_method', type=str, default='2d')


    # model arguments
    parser.add_argument('--chans', type=int, default=32)
    parser.add_argument('--cascades', type=int, default=6)
    parser.add_argument('--pi_sampling', action='store_true')
    parser.add_argument('--norm_all_k', action='store_true')
    parser.add_argument('--image_space_loss', type=str, default='')
    parser.add_argument('--k_loss', type=str, default='norml1l2')
    parser.add_argument('--image_loss_scaling', type=float, default=0)
    parser.add_argument('--split_contrast_by_phase', action='store_true')
    parser.add_argument('--sense_method', type=str, default='first')

    # loggin arguments
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--project', type=str, default='MRI Reconstruction')

    args = parser.parse_args()

    args = replace_args_from_config(args.config, args)


    main(args)
