from ml_recon.dataset.undersample_decorator import UndersampleDecorator
from ml_recon.utils import ifft_2d_img, root_sum_of_squares
from ml_recon.dataset.BraTS_dataset import BratsDataset
from ml_recon.dataset.m4raw_dataset import M4Raw
from ml_recon.dataset.fastMRI_dataset import FastMRIDataset
from ml_recon.dataset.dataset_output_type import TrainingSample

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import pytorch_lightning as pl

from dataclasses import asdict
import os


class UndersampledDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            dataset_name: str,
            data_dir: str, 
            batch_size: int, 
            R: float = 6,
            R_hat: float = 2.0,
            contrasts: list[str] = ['t1', 't1ce', 't2', 'flair'],
            resolution: tuple[int, int] = (128, 128),
            line_constrained: bool = True,
            num_workers: int = 0,
            norm_method: str = 'k',
            segregated: bool = False,
            self_supervsied: bool = False
            ):

        super().__init__()
        self.save_hyperparameters()
        
        dataset_name = str.lower(dataset_name)
        if dataset_name == 'brats': 
            self.dataset_class = BratsDataset
        elif dataset_name == 'fastmri':
            self.dataset_class = FastMRIDataset
        elif dataset_name == 'm4raw':
            self.dataset_class = M4Raw

        self.data_dir = data_dir
        self.contrasts = contrasts
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.resolution = resolution
        self.line_constrained = line_constrained
        self.segregated = segregated
        self.R = R
        self.R_hat = R_hat
        self.self_supervised = self_supervsied
        
        if norm_method == 'img':
            self.transforms = Compose([normalize_image_max(), convert_dataclass_to_dict()])
        else: 
            self.transforms = Compose([normalize_k_max(), convert_dataclass_to_dict()]) 

    def setup(self, stage):
        super().setup(stage)
        data_dir = os.listdir(self.data_dir)

        train_file = next((name for name in data_dir if 'train' in name))
        val_file = next((name for name in data_dir if 'val' in name))
        test_file = next((name for name in data_dir if 'test' in name))

        train_dir = os.path.join(self.data_dir, train_file)
        val_dir = os.path.join(self.data_dir, val_file)
        test_dir = os.path.join(self.data_dir, test_file)

        dataset_keyword_args = {
            'nx': self.resolution[0], 
            'ny': self.resolution[1],
            'contrasts': self.contrasts
        }

        undersample_keywords = {
                'R': self.R,
                'R_hat': self.R_hat,
                'line_constrained': self.line_constrained,
                'transforms': self.transforms,
                'segregated': self.segregated,
                'self_supervised': self.self_supervised
        }

        self.train_dataset = self.dataset_class(
                train_dir, 
                **dataset_keyword_args
                )

        self.val_dataset = self.dataset_class(
                val_dir, 
                **dataset_keyword_args
                )

        self.test_dataset = self.dataset_class(
                test_dir, 
                **dataset_keyword_args
                )

        self.train_dataset = UndersampleDecorator(
                self.train_dataset,
                **undersample_keywords
                )

        self.val_dataset = UndersampleDecorator(
                self.val_dataset,
                **undersample_keywords
                )
        
        self.test_dataset = UndersampleDecorator(
                self.test_dataset,
                **undersample_keywords
                )

        self.contrast_order = self.train_dataset.contrast_order
    
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
    def __call__(self, data: TrainingSample):
        input = data.input
        img = root_sum_of_squares(ifft_2d_img(input), coil_dim=1)
        scaling_factor = img.amax((1, 2), keepdim=True).unsqueeze(1)

        data.input /= scaling_factor
        data.target /= scaling_factor
        data.fs_k_space /= scaling_factor
        return data

class normalize_k_max(object):
    def __call__(self, data: TrainingSample):
        input = data.input
        undersample_max = input.abs().amax((1, 2, 3), keepdim=True)
        
        data.input /= undersample_max
        data.target /= undersample_max
        data.fs_k_space /= undersample_max
        return data

class convert_dataclass_to_dict(object):
    def __call__(self, data: TrainingSample):
        return asdict(data)
        