from argparse import ArgumentParser
import torch

from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
from ml_recon.utils import replace_args_from_config

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
    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         logger=wandb_logger, 
                         limit_train_batches=args.limit_batches,
                         limit_val_batches=args.limit_batches,
                         limit_test_batches=args.limit_batches,
                         precision="bf16-mixed", 
                         )


    data_dir = args.data_dir
    nx = args.nx
    ny = args.ny
    
    data_module = UndersampledDataModule(
            args.dataset, 
            data_dir, 
            batch_size=args.batch_size, 
            resolution=(ny, nx),
            num_workers=args.num_workers,
            contrasts=args.contrasts,
            line_constrained=args.line_constrained,
            pi_sampling=args.pi_sampling, 
            supervised_dataset=True,
            R=args.R
            ) 

    data_module.setup('train')
    

    model = LearnedSSLLightning(
            (len(args.contrasts), ny, nx), 
            inital_R=args.R_hat, 
            contrast_order=data_module.contrast_order,
            lr = args.lr,
            learn_R=args.learn_R,
            warm_start=args.warm_start,
            ssim_scaling_full=args.ssim_scaling_full,
            sigmoid_slope2=args.sigmoid_slope2,
            sigmoid_slope1=args.sigmoid_slope1,
            ssim_scaling_set=args.ssim_scaling_set,
            ssim_scaling_inverse=args.ssim_scaling_inverse,
            lambda_scaling=args.lambda_scaling,
            pass_all_data=args.pass_all_data,
            pass_inverse_data=args.pass_inverse_data,
            supervised=args.supervised,
            channels=args.chans,
            learn_sampling=args.learn_sampling,
            image_loss_function=args.image_loss,
            cascades=args.cascades, 
            warmup_training=args.warmup_training,
            k_space_loss_function=args.k_loss
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
    training_group.add_argument('--num_workers', type=int, default=0)
    training_group.add_argument('--max_epochs', type=int, default=50)
    training_group.add_argument('--batch_size', type=int, default=16)
    training_group.add_argument('--lr', type=float, default=1e-3)
    training_group.add_argument('--checkpoint', type=str)
    training_group.add_argument("--config", type=str, help="Path to the YAML configuration file.")
    
    # dataset parameters
    dataset_group = parser.add_argument_group('Dataset Parameters')
    dataset_group.add_argument('--R', type=float, default=6.0)
    dataset_group.add_argument('--dataset', type=str, default='brats')
    dataset_group.add_argument('--contrasts', type=str, nargs='+', default=['t1', 't2', 't1ce', 'flair'])
    dataset_group.add_argument('--nx', type=int, default=128)
    dataset_group.add_argument('--ny', type=int, default=128)
    dataset_group.add_argument('--limit_batches', type=float, default=1.0)
    dataset_group.add_argument('--line_constrained', action='store_true')
    dataset_group.add_argument('--pi_sampling', action='store_true')

    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--R_hat', type=float, default=2.0)
    model_group.add_argument('--learn_R', action='store_true')
    model_group.add_argument('--warm_start', action='store_true')
    model_group.add_argument('--data_dir', type=str, default='/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset_random_phase/')
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

    logger_group = parser.add_argument_group('Logging Parameters')
    logger_group.add_argument('--project', type=str, default='MRI Reconstruction')
    logger_group.add_argument('--run_name', type=str)
    
    args = parser.parse_args()

    args = replace_args_from_config(args.config, args)

    main(args)
