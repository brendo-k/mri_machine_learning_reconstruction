# python modules
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory
import os 
import re

# deep learning modules
import torch
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor

# my modules
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
from ml_recon.utils import replace_args_from_config, restore_optimizer
from ml_recon.pl_modules.pl_learn_ssl_undersampling import (
    LearnedSSLLightning, 
    VarnetConfig, 
    LearnPartitionConfig, 
    DualDomainConifg
    )

# pytorch lightning tools and trainers
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.callbacks import ModelCheckpoint 

if os.getenv('NORM_METHOD'):
    NORM_METHOD = os.getenv('NORM_METHOD') 
else:
    NORM_METHOD = 'image_mean'

if os.getenv('REDUCE_LOSS_BY_MASK'):
    REDUCE_LOSS_BY_MASK = bool(os.getenv('REDUCE_LOSS_BY_MASK'))
else:
    REDUCE_LOSS_BY_MASK = False
print(os.getenv('REDUCE_LOSS_BY_MASK'), REDUCE_LOSS_BY_MASK)

def main(args):
    pl.seed_everything(8, workers=True)
    file_name = get_unique_file_name(args)

    # build some callbacks for pytorch lightning
    callbacks = build_callbacks(args, file_name)
    
    # setup pytorch lightning dataloder and datamodules
    model, data_module = setup_model_and_dataloaders(args, callbacks)
    # setup wandb logger
    wandb_logger = setup_wandb_logger(args, model)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs, 
        logger=wandb_logger, 
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run
        )

    # use tensor cores
    torch.set_float32_matmul_precision('medium')

    trainer.fit(model=model, datamodule=data_module, ckpt_path=args.checkpoint)
    trainer.test(model, datamodule=data_module)

    process_checkpoint(args, callbacks, wandb_logger)

def process_checkpoint(args, callbacks, wandb_logger):
    checkpoint_path = Path(args.checkpoint_dir) / callbacks[0].best_model_path
    # log to wandb
    log_weights_to_wandb(wandb_logger, checkpoint_path)

def setup_model_and_dataloaders(args, callbacks):
    if args.checkpoint: 
        model, data_module = load_checkpoint(args, args.data_dir)
        callbacks.append(restore_optimizer(args.checkpoint))
    else:
        model, data_module = setup_model_parameters(args)
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

def setup_wandb_logger(args, model):
    if os.environ.get('SLURM_LOCALID') is None or int(os.environ['SLURM_LOCALID']) == 0:
        print(model.hparams)
        wandb_experiment = wandb.init(config=model.hparams, project=args.project, name=args.run_name, dir=args.logger_dir)
        logger = WandbLogger(experiment=wandb_experiment)
    else: 
        logger = DummyLogger()
    return logger
    

def setup_model_parameters(args):
    # setup model configurations

    data_module = UndersampledDataModule(
        args.dataset, 
        args.data_dir, 
        batch_size=args.batch_size, 
        resolution=(args.ny, args.nx),
        num_workers=args.num_workers,
        contrasts=args.contrasts,
        sampling_method=args.sampling_method,
        R=args.R,
        self_supervised=(not args.supervised), 
        ssdu_partitioning=args.ssdu_partitioning,
        limit_volumes=args.limit_volumes,
        same_mask_every_epoch=args.same_mask_all_epochs, 
        norm_method=NORM_METHOD,
        seed=8
    ) 
    data_module.setup('train')

    varnet_config = VarnetConfig(
        contrast_order=data_module.contrast_order,
        cascades=args.cascades, 
        channels=args.chans,
        depth=args.depth,
    )

    partitioning_config = LearnPartitionConfig(
        image_size=(len(args.contrasts), args.ny, args.nx),
        inital_R_value=args.R_hat,
        k_center_region = 10,
        sigmoid_slope_probability = args.sigmoid_slope1,
        sigmoid_slope_sampling = args.sigmoid_slope2,
        is_warm_start = args.warm_start,
        sampling_method = args.sampling_method,
        is_learn_R = args.learn_R,
        line_constrained=args.line_constrained

    )

    tripple_pathway_config = DualDomainConifg(
        is_pass_inverse=args.pass_inverse_data,
        is_pass_original=args.pass_all_data,
        inverse_no_grad=args.inverse_data_no_grad,
        original_no_grad=args.all_data_no_grad,
        pass_all_lines=args.pass_all_lines,
        pass_through_size=args.pass_through_size,
        seperate_models=args.seperate_model
    )

    model = LearnedSSLLightning(
        varnet_config = varnet_config, 
        learn_partitioning_config = partitioning_config, 
        dual_domain_config = tripple_pathway_config,
        lr = args.lr,
        image_loss_scaling_lam_full=args.image_scaling_lam_full,
        image_loss_scaling_lam_inv=args.image_scaling_lam_inv,
        image_loss_scaling_full_inv=args.image_scaling_full_inv,
        lambda_scaling=args.lambda_scaling, 
        image_loss_function=args.image_loss,
        k_space_loss_function=args.k_loss,
        enable_learn_partitioning=args.learn_sampling, 
        use_supervised_image_loss=args.supervised_image,
        enable_warmup_training=args.warmup_training,
        image_loss_grad_scaling=args.image_loss_grad_scaling,
    )

    return model, data_module

def log_weights_to_wandb(wandb_logger, checkpoint_path):
    # only run on the first rank if dataparallel job
    print(os.environ.get('SLURM_LOCALID'))
    if os.environ.get('SLURM_LOCALID'):
        rank = os.environ.get('SLURM_LOCALID')
    else:
        rank = 0

    if rank == 0:
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


def load_checkpoint(args, data_dir):
    print("Loading Checkpoint!")
    data_module_kwargs = {}
    if os.environ.get('SLURM_LOCALID') is not None:
        device = f"cuda:{int(os.environ['SLURM_LOCALID'])}"
    else:
        device = 'cpu' if torch.cuda.is_available() else 'cuda:0'
    if data_dir:
        data_module_kwargs['data_dir'] = data_dir
        data_module_kwargs['device'] = device
    model = LearnedSSLLightning.load_from_checkpoint(args.checkpoint, lr=args.lr, map_location=device)
    data_module = UndersampledDataModule.load_from_checkpoint(args.checkpoint, **data_module_kwargs)
    data_module.setup('train')
    return model, data_module

def build_checkpoint_callbacks(file_name, checkpoint_dir):
    if args.checkpoint: 
        file_name = Path(args.checkpoint).name
        file_name = re.sub(r'last-epoch=\d+\.ckpt$', '', file_name)
        checkpoint_dir = Path(args.checkpoint).parent
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
    if args.run_name and not args.checkpoint: 
        contrasts = ','.join(args.contrasts)
        file_name = f'{args.run_name}_{args.R}_{args.supervised}_{contrasts}_{unique_id}'
    if args.checkpoint: 
        file_name = f'{args.run_name}_{unique_id}'
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
    training_group.add_argument("--fast_dev_run", action='store_true')
    
    # dataset parameters
    dataset_group = parser.add_argument_group('Dataset Parameters')
    dataset_group.add_argument('--R', type=float, default=6.0)
    dataset_group.add_argument('--dataset', type=str, default='m4raw')
    dataset_group.add_argument('--contrasts', type=str, nargs='+', default=['t1', 't2', 'flair'])
    dataset_group.add_argument('--data_dir', type=str)
    dataset_group.add_argument('--nx', type=int, default=256)
    dataset_group.add_argument('--ny', type=int, default=256)
    dataset_group.add_argument('--limit_volumes', type=float, default=1.0)
    dataset_group.add_argument('--sampling_method', type=str, choices=['2d', '1d', 'pi'], default='2d')
    dataset_group.add_argument('--ssdu_partitioning', action='store_true')
    dataset_group.add_argument('--same_mask_all_epochs', action='store_true')

    # model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--R_hat', type=float, default=2.0)
    model_group.add_argument('--warm_start', action='store_true')
    model_group.add_argument('--chans', type=int, default=32)
    model_group.add_argument('--depth', type=int, default=4)
    model_group.add_argument('--cascades', type=int, default=6)
    model_group.add_argument('--sigmoid_slope2', type=float, default=200)
    model_group.add_argument('--sigmoid_slope1', type=float, default=5)
    model_group.add_argument('--pass_through_size', type=int, default=10)
    model_group.add_argument('--pass_all_lines', action='store_true')
    model_group.add_argument('--seperate_model', action='store_true')
    model_group.add_argument('--learn_R', action='store_true')

    # loss function parameters
    model_group.add_argument('--image_scaling_lam_inv', type=float, default=0.0)
    model_group.add_argument('--image_scaling_lam_full', type=float, default=0.0)
    model_group.add_argument('--image_scaling_full_inv', type=float, default=0.0)
    model_group.add_argument('--k_loss', type=str, default='l1', choices=['l1', 'l2', 'l1l2'])
    model_group.add_argument('--image_loss', type=str, default='ssim', choices=['ssim', 'l1_grad', 'l1'])
    model_group.add_argument('--image_loss_grad_scaling', type=float, default=1.)
    model_group.add_argument('--lambda_scaling', type=float, default=0.65)
    model_group.add_argument('--line_constrained', action='store_true')

    model_group.add_argument('--use_schedulers', action='store_true')
    model_group.add_argument('--norm_loss_by_mask', action='store_true')
    model_group.add_argument('--warmup_training', action='store_true')

    # configure pathways in triple pathway
    model_group.add_argument('--pass_inverse_data', action='store_true')
    model_group.add_argument('--pass_all_data', action='store_true')
    model_group.add_argument('--inverse_data_no_grad', action='store_true')
    model_group.add_argument('--all_data_no_grad', action='store_true')

    # training type (supervised, self-supervised)
    model_group.add_argument('--supervised', action='store_true')
    model_group.add_argument('--supervised_image', action='store_true')
    model_group.add_argument('--learn_sampling', action='store_true')
    
    #logging parameters
    logger_group = parser.add_argument_group('Logging Parameters')
    logger_group.add_argument('--project', type=str, default='MRI Reconstruction')
    logger_group.add_argument('--run_name', type=str)
    logger_group.add_argument('--logger_dir', type=str, default='/home/kadotab/scratch')
    
    args = parser.parse_args()

    args = replace_args_from_config(args.config, args, parser)

    main(args)
