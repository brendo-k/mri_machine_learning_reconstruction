import torch
from ml_recon.pl_modules.pl_varnet import pl_VarNet
   # Start recording memory snapshot history, initialized with a buffer
   # capacity of 100,000 memory events, via the `max_entries` field.
if __name__ == "__main__":
    torch.cuda.memory._record_memory_history(
        max_entries=100_000
    )

from argparse import ArgumentParser

from ml_recon.pl_modules.pl_varnet import pl_VarNet, VarnetConfig
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.profilers import PyTorchProfiler, AdvancedProfiler
from torch.profiler import ProfilerActivity, schedule
from datetime import datetime


"""
Training code for multicontrast dataset

This code takes a multi--contrast datset and trains a VarNet architceture

Examples:
    # Basic training
    train.py --num_workers 3 --max_epochs 50 --contrasts t1 t2 flair
"""
def main(args):
    data_dir = args.data_dir
    nx = args.nx
    ny = args.ny
    
    trainer = pl.Trainer(
                        max_steps=5,
                        logger=False
                     )
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
            supervised=args.supervised,
            pi_sampling=args.pi_sampling, 
            contrasts=args.contrasts, 
            ssdu_partioning=args.ssdu_partioning, 
            ) 

    data_module.setup('train')

    model_config = VarnetConfig(
        model_name=args.model, 
        contrast_order=data_module.contrast_order,
        lr = args.lr, 
        cascades=args.cascades, 
        channels=args.chans,
        norm_all_k=args.norm_all_k,
        image_loss_function=args.image_space_loss,
        image_loss_scaling=args.image_loss_scaling
    )
    
    model = LearnedSSLLightning(
            (len(args.contrasts), ny, nx), 
            R_parameter=args.R_hat, 
            contrast_order=data_module.contrast_order,
            lr = args.lr,
            supervised=False,
            channels=args.chans,
            learn_sampling=False,
            cascades=args.cascades, 
            )

    # model = pl_VarNet(
    #             config=model_config
    #         )

    trainer.fit(model, datamodule=data_module)
    try:
        torch.cuda.memory._dump_snapshot(f"mem2.pickle")
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")
        return 

    # Stop recording memory snapshot history.
    torch.cuda.memory._record_memory_history()
    
    
if __name__ == '__main__': 
    parser = ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--line_constrained', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--R', type=float, default=4.0)
    parser.add_argument('--R_hat', type=float, default=2.0)
    parser.add_argument('--limit_batches', type=float, default=1.0)
    parser.add_argument('--nx', type=int, default=240)
    parser.add_argument('--ny', type=int, default=240)
    parser.add_argument('--norm_method', type=str, default='k')
    parser.add_argument('--supervised', action='store_true')
    parser.add_argument('--data_dir', type=str, default='/home/brenden/Documents/data/simulated_subset_random_phase')
    parser.add_argument('--contrasts', type=str, nargs='+', default=['t1', 't2', 't1ce', 'flair'])
    parser.add_argument('--chans', type=int, default=18)
    parser.add_argument('--cascades', type=int, default=3)
    parser.add_argument('--dataset_name', type=str, default='brats')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--project', type=str, default='MRI Reconstruction')
    parser.add_argument('--pi_sampling', action='store_false')
    parser.add_argument('--ssdu_partioning', action='store_true')
    parser.add_argument('--norm_all_k', action='store_true')
    parser.add_argument('--image_space_loss', type=str, default='')
    parser.add_argument('--image_loss_scaling', type=float, default=0)
    
    args = parser.parse_args()

    main(args)