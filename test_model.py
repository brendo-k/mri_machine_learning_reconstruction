import os
import argparse

from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning

import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger


def main(args):
    pl.seed_everything(8)
    data_dir = args.data_dir
    test_dir = args.test_dir
    checkpoint_path = args.checkpoint
    logger = WandbLogger(project=args.project, name=args.run_name)
    if args.wandb_artifact:
        artifact = logger.use_artifact(args.wandb_artifact)
        artifact_dir = artifact.download()
        checkpoint_path = os.path.join(artifact_dir, 'model.ckpt')
    
    return test(checkpoint_path=checkpoint_path, data_dir=data_dir, test_dir=test_dir, logger=logger)

def test(checkpoint_path, data_dir=None, test_dir=None, logger=None, is_mask_testing=True, mask_threshold=None, batch_size=1):
    # Load model and data module
    args = {}
    model = LearnedSSLLightning.load_from_checkpoint(
        checkpoint_path,
        is_mask_testing = is_mask_testing,
        mask_theshold = mask_threshold,
        )

    data_module_kwargs = {}
    if test_dir: 
        data_module_kwargs['test_dir'] = test_dir
    if data_dir:
        data_module_kwargs['data_dir'] = data_dir

    datamodule = UndersampledDataModule.load_from_checkpoint(checkpoint_path, batch_size=batch_size, **data_module_kwargs)

    # Test model
    trainer = pl.Trainer(logger=logger, accelerator='cuda')
    metrics = trainer.test(model, datamodule=datamodule)
    return metrics


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--wandb_artifact', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--run_name', type=str)

    
    args = parser.parse_args()    
    main(args)
