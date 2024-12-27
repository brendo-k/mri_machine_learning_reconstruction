from argparse import ArgumentParser
import torch

from ml_recon.pl_modules.pl_learn_ssl_undersampling import (
    LearnedSSLLightning, 
    VarnetConfig, 
    LearnPartitionConfig, 
    DualDomainConifg
    )
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
from ml_recon.models.MultiContrastVarNet import VarnetConfig
from ml_recon.utils import replace_args_from_config
from pytorch_lightning.profilers import PyTorchProfiler, AdvancedProfiler
from pytorch_lightning.tuner.tuning import Tuner
from torch.profiler import ProfilerActivity, schedule

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from datetime import datetime

def main(args):
    pl.seed_everything(8)
    wandb_logger = WandbLogger(project=args.project, log_model=True, name=args.run_name,)
    unique_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = 'pl_learn_ssl-' + unique_id
    #checkpoint_callback = ModelCheckpoint(
    #    dirpath='checkpoints/',  # Directory to save the checkpoints
    #    filename=file_name + '-{epoch:02d}-{val_loss:.2f}',  # Filename pattern
    #    save_top_k=1,  # Save the top 3 models
    #    monitor='val/val_loss_lambda',  # Metric to monitor for saving the best models
    #    mode='min',  # Save the model with the minimum val_loss
    #)
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
    advanced_prof = AdvancedProfiler(
        dirpath='.', filename='prof'
    )
    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         logger=wandb_logger, 
                         limit_train_batches=args.limit_batches,
                         limit_val_batches=args.limit_batches,
                         limit_test_batches=args.limit_batches,
                         #precision="bf16-mixed", 
                         #profiler=pytorch_profiler
                         )


    data_dir = args.data_dir
    nx = args.nx
    ny = args.ny
    
    data_module = UndersampledDataModule(
            args.dataset, 
            data_dir, 
            args.test_dir,
            batch_size=args.batch_size, 
            resolution=(ny, nx),
            num_workers=args.num_workers,
            contrasts=args.contrasts,
            sampling_method=args.sampling_method,
            R=args.R,
            self_supervsied=(not args.supervised)
            ) 

    data_module.setup('train')

    
    varnet_config = VarnetConfig(
        contrast_order=data_module.contrast_order,
        cascades=args.cascades, 
        channels=args.chans,
        split_contrast_by_phase=args.split_contrast_by_phase,
        sensetivity_estimation=args.sense_method

    )
    
    partitioning_config = LearnPartitionConfig(
        image_size=(len(args.contrasts), args.ny, args.nx),
        inital_R_value=args.R_hat,
        k_center_region = 10,
        sigmoid_slope_probability = args.sigmoid_slope1,
        sigmoid_slope_sampling = args.sigmoid_slope2,
        is_learn_R = args.learn_R,
        is_warm_start = args.warm_start,
        
    )

    tripple_pathway_config = DualDomainConifg(
        is_pass_inverse=args.pass_inverse_data,
        is_pass_original=args.pass_all_data
    )

    model = LearnedSSLLightning(
        varnet_config = varnet_config, 
        learn_partitioning_config = partitioning_config, 
        dual_domain_config = tripple_pathway_config,
        lr = args.lr,
        ssim_scaling_full=args.ssim_scaling_full,
        ssim_scaling_set=args.ssim_scaling_set,
        ssim_scaling_inverse=args.ssim_scaling_inverse,
        lambda_scaling=args.lambda_scaling, 
        image_loss_function=args.image_loss,
        k_space_loss_function=args.k_loss,
        is_learn_partitioning=args.learn_sampling, 
        is_norm_loss = args.norm_loss_by_masks,
        )
    torch.set_float32_matmul_precision('medium')

    if args.checkpoint: 
        print("Loading Checkpoint!")
        model = LearnedSSLLightning.load_from_checkpoint(args.checkpoint)
        data_module = UndersampledDataModule.load_from_checkpoint(args.checkpoint)
        data_module.setup('train')

    ## AUTOMATIC HYPERPARAMETER TUNING
    #tuner = Tuner(trainer)
    #tuner.scale_batch_size(model, mode='binsearch', datamodule=data_module)
    #tuner.lr_find(model, datamodule=data_module, min_lr=1e-4, max_lr=1e-1)

    #wandb_logger.experiment.config.update(model.hparams)

    print(model.hparams)
    trainer.fit(model=model, datamodule=data_module, ckpt_path=args.checkpoint)
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__': 
    parser = ArgumentParser(description="Deep learning multi-contrast reconstruction")

    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--num_workers', type=int, default=3)
    training_group.add_argument('--max_epochs', type=int, default=50)
    training_group.add_argument('--batch_size', type=int, default=1)
    training_group.add_argument('--lr', type=float, default=1e-3)
    training_group.add_argument('--checkpoint', type=str)
    training_group.add_argument("--config", type=str, help="Path to the YAML configuration file.")
    
    # dataset parameters
    dataset_group = parser.add_argument_group('Dataset Parameters')
    dataset_group.add_argument('--R', type=float, default=6.0)
    dataset_group.add_argument('--dataset', type=str, default='m4raw')
    dataset_group.add_argument('--contrasts', type=str, nargs='+', default=['t1', 't2', 'flair'])
    dataset_group.add_argument('--data_dir', type=str, default="/Users/brend/Documents/Data")
    dataset_group.add_argument('--test_dir', type=str, default="/Users/brend/Documents/Data")
    dataset_group.add_argument('--nx', type=int, default=256)
    dataset_group.add_argument('--ny', type=int, default=256)
    dataset_group.add_argument('--limit_batches', type=float, default=1.0)
    dataset_group.add_argument('--sampling_method', type=str, choices=['2d', '1d', 'pi'], default='2d')


    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--R_hat', type=float, default=2.0)
    model_group.add_argument('--learn_R', action='store_true')
    model_group.add_argument('--warm_start', action='store_true')
    model_group.add_argument('--chans', type=int, default=32)
    model_group.add_argument('--cascades', type=int, default=6)
    model_group.add_argument('--ssim_scaling_full', type=float, default=0.0)
    model_group.add_argument('--ssim_scaling_set', type=float, default=0.0)
    model_group.add_argument('--ssim_scaling_inverse', type=float, default=0.0)
    model_group.add_argument('--lambda_scaling', type=float, default=1)
    model_group.add_argument('--sigmoid_slope2', type=float, default=200)
    model_group.add_argument('--sigmoid_slope1', type=float, default=5)
    model_group.add_argument('--pass_inverse_data', action='store_true')
    model_group.add_argument('--pass_all_data', action='store_true')
    model_group.add_argument('--learn_sampling', action='store_true')
    model_group.add_argument('--supervised', action='store_true')
    model_group.add_argument('--image_loss', type=str, default='ssim')
    model_group.add_argument('--k_loss', type=str, default='l1l2')
    model_group.add_argument('--warmup_training', action='store_true')
    model_group.add_argument('--split_contrast_by_phase', action='store_true')
    model_group.add_argument('--sense_method', type=str, default='first')
    model_group.add_argument('--norm_loss_by_masks', action='store_true')

    logger_group = parser.add_argument_group('Logging Parameters')
    logger_group.add_argument('--project', type=str, default='MRI Reconstruction')
    logger_group.add_argument('--run_name', type=str)
    
    args = parser.parse_args()

    args = replace_args_from_config(args.config, args)

    main(args)
