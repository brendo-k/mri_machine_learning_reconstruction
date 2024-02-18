import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ml_recon.dataset.k_space_dataset import KSpaceDataset
from ml_recon.dataset.Brats_dataset import BratsDataset
from ml_recon.dataset.fastMRI_dataset import FastMRIDataset
from ml_recon.utils import root_sum_of_squares, ifft_2d_img

class MRI_Loader(pl.LightningDataModule):
    def __init__(
            self, 
            dataset_name: str, 
            data_dir: str, 
            resolution: tuple[int, int] = (128, 128),
            contrasts: list[str] = ['t1', 't1ce', 't2', 'flair'],
            num_workers: int = 0,
            batch_size: int = 4
            ):

        super().__init__()

        self.data_dir = data_dir 
        self.batch_size = batch_size
        self.resolution = resolution
        self.contrasts = contrasts
        self.num_workers = num_workers

        if dataset_name == 'brats': 
            self.dataset_class = BratsDataset
        elif dataset_name == 'fastmri':
            self.dataset_class = FastMRIDataset

    def setup(self, stage):
        data_dir = os.listdir(self.data_dir)
        print(self.contrasts)

        train_file = next((name for name in data_dir if 'train' in name))
        val_file = next((name for name in data_dir if 'val' in name))
        test_file = next((name for name in data_dir if 'test' in name))

        train_dir = os.path.join(self.data_dir, train_file)
        val_dir = os.path.join(self.data_dir, val_file)
        test_dir = os.path.join(self.data_dir, test_file)

        self.train_dataset = self.dataset_class(train_dir, nx=self.resolution[0], ny=self.resolution[1], contrasts=self.contrasts)
        self.val_dataset = self.dataset_class(val_dir, nx=self.resolution[0], ny=self.resolution[1], contrasts=self.contrasts)
        self.test_dataset = self.dataset_class(test_dir, nx=self.resolution[0], ny=self.resolution[1], contrasts=self.contrasts)

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True
                )

    def val_dataloader(self):
        return DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                pin_memory=True
                )

    def test_dataloader(self):
        return DataLoader(
                self.test_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                pin_memory=True
                )


class normalize_image_max(object):
    def __call__(self, data):
        target = data
        img = root_sum_of_squares(ifft_2d_img(target), coil_dim=1)

        return target/img.max()

class normalize_k_max(object):
    def __call__(self, data):
        undersampled, target = data
        k_max = target.abs().max()

        return target/k_max

