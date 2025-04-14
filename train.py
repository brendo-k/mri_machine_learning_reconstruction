from argparse import ArgumentParser
import torch
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor
import matplotlib.pyplot as plt
from pathlib import Path

from tempfile import TemporaryDirectory

from ml_recon.pl_modules.pl_learn_ssl_undersampling import (
    LearnedSSLLightning, 
    VarnetConfig, 
    LearnPartitionConfig, 
    DualDomainConifg
    )

from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
from ml_recon.models.MultiContrastVarNet import VarnetConfig
from ml_recon.utils import replace_args_from_config, restore_optimizer


import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner
from datetime import datetime

def main(args):
    pl.seed_everything(8)
    file_name = get_unique_file_name(args)

    # build some callbacks for pytorch lightning
    callbacks = build_callbacks(args, file_name)
    
    # setup threshold for masking background
    thresholds = setup_masking_thresholds(args)
    
    # setup pytorch lightning dataloder and datamodules
    model, data_module = setup_model_and_dataloaders(args, callbacks, thresholds)
    # setup wandb logger
    wandb_logger = setup_wandb_logger(args, model, data_module)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs, 
        logger=wandb_logger, 
        callbacks=callbacks,
        )

    tuner = Tuner(trainer)

    torch.set_float32_matmul_precision('medium')
    trainer.fit(model=model, datamodule=data_module, ckpt_path=args.checkpoint)
    trainer.test(model, datamodule=data_module)

    process_checkpoint(args, callbacks, wandb_logger)

def process_checkpoint(args, callbacks, wandb_logger):
    checkpoint_path = Path(args.checkpoint_dir) / callbacks[0].best_model_path
    # log to wandb
    log_weights_to_wandb(wandb_logger, checkpoint_path)

def setup_model_and_dataloaders(args, callbacks, thresholds):
    if args.checkpoint: 
        model, data_module = load_checkpoint(args, args.data_dir, args.test_dir)
        callbacks.append(restore_optimizer(args.checkpoint))
    else:
        model, data_module = setup_model_parameters(args, thresholds)
    return model,data_module

def build_callbacks(args, file_name):
    callbacks = []
    callbacks.append(build_checkpoint_callbacks(file_name, args.checkpoint_dir))
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    return callbacks

def restore_optimizer_state(model):
    optim = model.optimizers()
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    optim.load_state_dict(checkpoint['state_dict'])

def setup_wandb_logger(args, model, data_module):
    hparams = dict(model.hparams)
    hparams.update(data_module.hparams)
    #config = vars(args)

    wandb_experiment = wandb.init(config=hparams, project=args.project, name=args.run_name, dir=args.logger_dir)
    wandb_logger = WandbLogger(experiment=wandb_experiment)
    wandb.define_metric("trainer/global_step")
    wandb.define_metric("*", step_metric="trainer/global_step")
    return wandb_logger


def setup_masking_thresholds(args):
    possible_contrasts = ['t1', 't2', 'flair', 't1ce'] 
    thresholds = {}
    for contrast in possible_contrasts:
        threshold = getattr(args, f'mask_threshold_{contrast}')
        if threshold is not None:
            thresholds[contrast] = threshold
    if len(thresholds) == 0: 
        thresholds = None
    return thresholds

def setup_model_parameters(args, thresholds):
    data_module = UndersampledDataModule(
                args.dataset, 
                args.data_dir, 
                args.test_dir,
                batch_size=args.batch_size, 
                resolution=(args.ny, args.nx),
                num_workers=args.num_workers,
                contrasts=args.contrasts,
                sampling_method=args.sampling_method,
                R=args.R,
                self_supervsied=(not args.supervised), 
                ssdu_partioning=args.ssdu_partitioning,
                acs_lines=args.acs_lines,
                norm_method=args.norm_method,
                limit_volumes=args.limit_volumes
                ) 
    data_module.setup('train')
        
    varnet_config = VarnetConfig(
            contrast_order=data_module.contrast_order,
            cascades=args.cascades, 
            channels=args.chans,
            depth=args.depth,
            sensetivity_estimation=args.sense_method,
            dropout=args.dropout, 
            upsample_method=args.upsample_method, 
            conv_after_upsample=args.conv_after_upsample
        )
        
    partitioning_config = LearnPartitionConfig(
            image_size=(len(args.contrasts), args.ny, args.nx),
            inital_R_value=args.R_hat,
            k_center_region = 10,
            sigmoid_slope_probability = args.sigmoid_slope1,
            sigmoid_slope_sampling = args.sigmoid_slope2,
            is_warm_start = args.warm_start,
            sampling_method = args.sampling_method
        )

    tripple_pathway_config = DualDomainConifg(
            is_pass_inverse=args.pass_inverse_data,
            is_pass_original=args.pass_all_data,
            inverse_no_grad=args.inverse_data_no_grad,
            original_no_grad=args.all_data_no_grad
        )

    model = LearnedSSLLightning(
            varnet_config = varnet_config, 
            learn_partitioning_config = partitioning_config, 
            dual_domain_config = tripple_pathway_config,
            lr = args.lr,
            lr_scheduler = args.lr_scheduler,
            warmup_adam = args.warmup_adam,
            image_loss_scaling_lam_full=args.ssim_scaling_full + args.ssim_scaling_delta,
            image_loss_scaling_lam_inv=args.ssim_scaling_set + args.ssim_scaling_delta,
            image_loss_scaling_full_inv=args.ssim_scaling_inverse + args.ssim_scaling_delta,
            lambda_scaling=args.lambda_scaling, 
            image_loss_function=args.image_loss,
            image_loss_l1_grad_scaling=args.l1_grad_scaling,
            k_space_loss_function=args.k_loss,
            enable_learn_partitioning=args.learn_sampling, 
            use_supervised_image_loss=args.supervised_image,
            weight_decay=args.weight_decay,
            pass_through_size=args.pass_through_size,
            mask_theshold=thresholds,
            enable_warmup_training=args.warmup_training,
            normalize_loss_by_mask=args.norm_loss_by_mask,
            )
        
    return model,data_module

def log_weights_to_wandb(wandb_logger, checkpoint_path):
    checkpoint_name = f"model-{wandb_logger.experiment.id}"

    # always remove optimizer state when logging to wandb
    with TemporaryDirectory() as tempdir: 
        temp_checkpoint = Path(tempdir) / 'model.ckpt'
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        torch.save(checkpoint, temp_checkpoint.as_posix())
        remove_optimizer_state(temp_checkpoint)

        artifact = wandb.Artifact(name=checkpoint_name, type="model")
        artifact.add_file(local_path=temp_checkpoint.as_posix(), name='model.ckpt')
        wandb_logger.experiment.log_artifact(artifact, aliases=['latest'])


def remove_optimizer_state(checkpoint_path, ):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if 'optimizer_states' in checkpoint:
        del checkpoint['optimizer_states']
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(args, data_dir, test_dir):
    print("Loading Checkpoint!")
    model = LearnedSSLLightning.load_from_checkpoint(args.checkpoint, lr=args.lr)
    data_module = UndersampledDataModule.load_from_checkpoint(args.checkpoint, data_dir=data_dir, test_dir=test_dir)
    data_module.setup('train')
    return model, data_module

def build_checkpoint_callbacks(file_name, checkpoint_dir):
    # Checkpoint for the last model (including optimizer state for resubmittions)
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=file_name + '-last-{epoch:02d}',
        save_top_k=1,  # Only keep the latest model
        monitor='epoch',
        mode='max',
        save_weights_only=False  # Save full model state including optimizer
    )
    
    return last_checkpoint_callback

def get_unique_file_name(args):
    unique_id = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    file_name = unique_id
    if args.run_name: 
        contrasts = ','.join(args.contrasts)
        file_name = f'{args.run_name}_{args.R}_{args.supervised}_{contrasts}_{unique_id}'
    return file_name


if __name__ == '__main__': 
    parser = ArgumentParser(description="Deep learning multi-contrast self-supervised reconstruction")

    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--num_workers', type=int, default=3)
    training_group.add_argument('--max_epochs', type=int, default=50)
    training_group.add_argument('--batch_size', type=int, default=1)
    training_group.add_argument('--lr', type=float, default=1e-3)
    training_group.add_argument('--lr_scheduler', action='store_true') 
    training_group.add_argument('--warmup_adam', action='store_true') 
    training_group.add_argument('--checkpoint', type=str)
    training_group.add_argument("--config", "-c", type=str, help="Path to the YAML configuration file.")
    training_group.add_argument("--checkpoint_dir", type=str, default='./checkpoints', help="Path to checkpoint save dir")
    
    # dataset parameters
    dataset_group = parser.add_argument_group('Dataset Parameters')
    dataset_group.add_argument('--R', type=float, default=6.0)
    dataset_group.add_argument('--acs_lines', type=int, default=10)
    dataset_group.add_argument('--dataset', type=str, default='m4raw')
    dataset_group.add_argument('--contrasts', type=str, nargs='+', default=['t1', 't2', 'flair'])
    dataset_group.add_argument('--data_dir', type=str, default="/Users/brend/Documents/Data")
    dataset_group.add_argument('--test_dir', type=str, default="/Users/brend/Documents/Data")
    dataset_group.add_argument('--nx', type=int, default=256)
    dataset_group.add_argument('--ny', type=int, default=256)
    dataset_group.add_argument('--limit_volumes', type=float, default=1.0)
    dataset_group.add_argument('--sampling_method', type=str, choices=['2d', '1d', 'pi'], default='2d')
    dataset_group.add_argument('--ssdu_partitioning', action='store_true')
    dataset_group.add_argument('--norm_method', default='k')


    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--R_hat', type=float, default=2.0)
    model_group.add_argument('--warm_start', action='store_true')
    model_group.add_argument('--chans', type=int, default=32)
    model_group.add_argument('--depth', type=int, default=4)
    model_group.add_argument('--cascades', type=int, default=6)
    model_group.add_argument('--sigmoid_slope2', type=float, default=200)
    model_group.add_argument('--sigmoid_slope1', type=float, default=5)
    model_group.add_argument('--pass_through_size', type=int, default=10)
    model_group.add_argument('--dropout', type=float, default=0.0) 
    model_group.add_argument('--weight_decay', type=float, default=0.0) 
    model_group.add_argument('--conv_after_upsample', action='store_true') 
    model_group.add_argument('--upsample_method', type=str, default='conv') 

    model_group.add_argument('--ssim_scaling_set', type=float, default=0.0)
    model_group.add_argument('--ssim_scaling_full', type=float, default=0.0)
    model_group.add_argument('--ssim_scaling_inverse', type=float, default=0.0)
    model_group.add_argument('--ssim_scaling_delta', type=float, default=0.0)
    model_group.add_argument('--k_loss', type=str, default='l1l2', choices=['l1', 'l2', 'l1l2'])
    model_group.add_argument('--image_loss', type=str, default='ssim', choices=['ssim', 'l1_grad', 'l1'])
    model_group.add_argument('--l1_grad_scaling', type=float, default=1)
    model_group.add_argument('--lambda_scaling', type=float, default=1)

    model_group.add_argument('--warmup_training', action='store_true')
    model_group.add_argument('--use_schedulers', action='store_true')
    model_group.add_argument('--sense_method', type=str, default='first')
    model_group.add_argument('--norm_loss_by_mask', action='store_true')
    model_group.add_argument('--pass_inverse_data', action='store_true')
    model_group.add_argument('--pass_all_data', action='store_true')
    model_group.add_argument('--inverse_data_no_grad', action='store_true')
    model_group.add_argument('--all_data_no_grad', action='store_true')
    model_group.add_argument('--learn_sampling', action='store_true')
    model_group.add_argument('--mask_threshold_t2', type=float)
    model_group.add_argument('--mask_threshold_t1', type=float)
    model_group.add_argument('--mask_threshold_flair', type=float)
    model_group.add_argument('--mask_threshold_t1ce', type=float)

    model_group.add_argument('--supervised', action='store_true')
    model_group.add_argument('--supervised_image', action='store_true')

    logger_group = parser.add_argument_group('Logging Parameters')
    logger_group.add_argument('--project', type=str, default='MRI Reconstruction')
    logger_group.add_argument('--run_name', type=str)
    logger_group.add_argument('--logger_dir', type=str, default='/home/kadotab/scratch')
    
    args = parser.parse_args()

    args = replace_args_from_config(args.config, args, parser)

    main(args)
