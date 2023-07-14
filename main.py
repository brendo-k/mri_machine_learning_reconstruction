import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from ml_recon.pl_modules.pl_varnet import pl_VarNet
from ml_recon.pl_modules.data_module import fastMRIDataLoader
from ml_recon.transforms import pad_recon, pad, toTensor

def cli_main():
    cli = LightningCLI(pl_VarNet, fastMRIDataLoader)

if __name__ == '__main__':
    cli_main()