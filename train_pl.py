from ml_recon.Models.pl_modules.pl_varnet import pl_VarNet
from ml_recon.Dataset.undersampled_dataset import UndersampledKSpaceDataset
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from ml_recon.Utils.collate_function import collate_fn
from torchvision.transforms import Compose
from ml_recon.Transforms import (toTensor, pad, normalize)

transforms = Compose(
    (
        pad((640, 320)), 
        toTensor(),
        normalize(),
    )
)

path = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/fastMRI/multicoil_train'
dataset = UndersampledKSpaceDataset(path, transforms=transforms, R=4)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

net = pl_VarNet(2, 2)

Trainer = pl.Trainer(accelerator="gpu",  max_epochs=7)
Trainer.fit(model=net, train_dataloaders=dataloader)