import os
from typing import Callable, Union

import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ml_recon.transforms import normalize
from ml_recon.dataset.sliceloader import SliceLoader


class fastMRIDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_dir: Union[str, os.PathLike], 
            nx: int = 256,
            ny: int = 256, 
            acs_lines: int = 10, 
            poly_order: int = 8, 
            R: int = 4, 
            R_hat: int = 2, 
            batch_size: int = 5, 
            transforms: Callable = None,
            num_workers: int = 0
            ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms
        self.num_workers = num_workers
        self.kargs = {
                'R': R, 
                'R_hat': R_hat, 
                'nx': nx, 
                'ny': ny, 
                'acs_lines': acs_lines, 
                'poly_order': poly_order,
                }

        # transforms, convert to tensor and normalize
        self.transform = normalize()

    def prepare_data(self):
        train_dir = os.path.join(self.data_dir, 'multicoil_train')
        test_dir = os.path.join(self.data_dir, 'multicoil_test')
        val_dir = os.path.join(self.data_dir, 'multicoil_val')
        
        # if supervised use supervised loader
        self.fastmri_train = SliceLoader(train_dir, **self.kargs)
        self.fastmri_val = SliceLoader(test_dir, **self.kargs)
        self.fastmri_test = SliceLoader(val_dir, **self.kargs)

    def train_dataloader(self):
        return DataLoader(
                self.fastmri_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
                )

    def val_dataloader(self):
        return DataLoader(
                self.fastmri_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers
                )

    def test_dataloader(self):
        return DataLoader(
                self.fastmri_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers
                )
