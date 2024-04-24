from argparse import ArgumentParser
from pytorch_lightning.loggers.wandb import WandbLogger

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--artifact_name', type=str)
    args = parser.parse_args()

    artifact = WandbLogger.download_artifact(args.artifact_name, artifact_type='model')
    print(artifact)
