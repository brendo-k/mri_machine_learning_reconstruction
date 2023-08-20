import os
from typing import Callable, Union

import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ml_recon.transforms import to_tensor, normalize
from ml_recon.dataset.sliceloader import SliceLoader
from ml_recon.dataset.self_supervised_slice_loader import SelfSupervisedSampling
from ml_recon.utils.read_headers import make_header


class fastMRIDataLoader(pl.LightningDataModule):
    def __init__(
            self, 
            data_dir: Union[str, os.PathLike], 
            supervised: bool = True,
            batch_size: int = 1, 
            transforms: Callable = None,
            num_workers: int = 0
            ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms
        self.num_workers = num_workers

        self.supervised = supervised

        # transforms, convert to tensor and normalize
        self.transform = torchvision.transforms.Compose((to_tensor(), normalize()))

    def prepare_data(self):
        train_dir = os.path.join(self.data_dir, 'multicoil_train')
        test_dir = os.path.join(self.data_dir, 'multicoil_test')
        val_dir = os.path.join(self.data_dir, 'multicoil_val')
        if not self.header_exists(test_dir):
            make_header(test_dir)
        if not self.header_exists(train_dir):
            make_header(train_dir)
        if not self.header_exists(val_dir):
            make_header(val_dir)
        
        if self.supervised:
            # if supervised use supervised loader
            self.fastmri_train = SliceLoader(train_dir + 'header.json')
            self.fastmri_val = SliceLoader(val_dir + 'header.json')
            self.fastmri_test = SliceLoader(test_dir + 'header.json')
        else:
            # else use self supervised loader
            self.fastmri_train = SelfSupervisedSampling(train_dir + 'header.json')
            self.fastmri_val = SelfSupervisedSampling(val_dir + 'header.json')
            self.fastmri_test = SelfSupervisedSampling(test_dir + 'header.json')

    def train_dataloader(self):
        return DataLoader(self.fastmri_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.fastmri_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.fastmri_test, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def header_exists(self, path):
        assert os.path.isdir(path)
        return os.path.isfile(path + 'header.json')