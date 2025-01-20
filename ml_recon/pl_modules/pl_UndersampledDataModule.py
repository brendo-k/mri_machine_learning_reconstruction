from ml_recon.dataset.undersample_decorator import UndersampleDecorator
from ml_recon.utils import ifft_2d_img, root_sum_of_squares, fft_2d_img
from ml_recon.dataset.BraTS_dataset import BratsDataset
from ml_recon.dataset.BraTS_test_dataset import BratsDatasetTest
from ml_recon.dataset.M4Raw_dataset import M4Raw
from ml_recon.dataset.M4Raw_test_dataset import M4RawTest
from ml_recon.dataset.FastMRI_dataset import FastMRIDataset
from ml_recon.dataset.FastMRI_test_dataset import FastMRIDatasetTest

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
            test_dir: str,
            batch_size: int, 
            R: float = 6,
            R_hat: float = 2.0,
            contrasts: list[str] = ['t1', 't1ce', 't2', 'flair'],
            resolution: tuple[int, int] = (128, 128),
            num_workers: int = 0,
            norm_method: str = 'k',
            self_supervsied: bool = False,
            sampling_method: str = '2d',
            ssdu_partioning: bool = False,
            acs_lines: int = 10
            ):

        super().__init__()
        self.save_hyperparameters()
        
        dataset_name = str.lower(dataset_name)
        if dataset_name == 'brats': 
            self.dataset_class = BratsDataset
            self.test_dataset_class = BratsDatasetTest
        elif dataset_name == 'fastmri':
            self.dataset_class = FastMRIDataset
            self.test_dataset_class = FastMRIDatasetTest
        elif dataset_name == 'm4raw':
            self.dataset_class = M4Raw
            self.test_dataset_class = M4RawTest

        self.data_dir = data_dir
        self.test_dir = test_dir
        self.contrasts = contrasts
        self.acs_lines = acs_lines
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.resolution = resolution
        self.R = R
        self.R_hat = R_hat
        self.self_supervised = self_supervsied
        self.ssdu_partioning = ssdu_partioning
        self.sampling_method = sampling_method
        self.norm_method = norm_method
        
        if norm_method == 'img':
            self.transforms = normalize_image_max()
        elif norm_method == 'k': 
            self.transforms = normalize_k_max() 
        elif norm_method == 'image_mean':
            self.transforms = normalize_image_mean() 
        elif norm_method == 'image_mean2':
            self.transforms = normalize_image_mean2() 

    def setup(self, stage):
        super().setup(stage)
        data_dir = os.listdir(self.data_dir)

        train_file = next((name for name in data_dir if 'train' in name))
        val_file = next((name for name in data_dir if 'val' in name))
        test_file = next((name for name in data_dir if 'test' in name))

        train_dir = os.path.join(self.data_dir, train_file)
        val_dir = os.path.join(self.data_dir, val_file)
        test_dir = os.path.join(self.data_dir, test_file)
        test_img_dir = os.path.join(self.test_dir, test_file)

        dataset_keyword_args = {
            'nx': self.resolution[0], 
            'ny': self.resolution[1],
            'contrasts': self.contrasts
        }

        undersample_keyword_args = {
                'R': self.R,
                'R_hat': self.R_hat,
                'sampling_method': self.sampling_method,
                'self_supervised': self.self_supervised,
                'acs_lines' : self.acs_lines
        }

        self.train_dataset = self.dataset_class(
                train_dir, 
                **dataset_keyword_args
                )

        self.val_dataset = self.dataset_class(
                val_dir, 
                **dataset_keyword_args
                )

        self.train_dataset = UndersampleDecorator(
                self.train_dataset,
                original_ssdu_partioning=self.ssdu_partioning,
                transforms=self.transforms,
                **undersample_keyword_args
                )

        self.val_dataset = UndersampleDecorator(
                self.val_dataset,
                original_ssdu_partioning=self.ssdu_partioning,
                transforms=self.transforms,
                **undersample_keyword_args
                )

        self.test_dataset = self.test_dataset_class(
                test_dir,
                test_img_dir,
                transforms=test_transform(),
                **dataset_keyword_args,
                **undersample_keyword_args
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
    def __call__(self, data: dict):
        input = data['fs_k_space']
        img = root_sum_of_squares(ifft_2d_img(input), coil_dim=1)
        scaling_factor = img.amax((1, 2), keepdim=True).unsqueeze(1)

        data['undersampled'] /= scaling_factor
        data['fs_k_space'] /= scaling_factor
        return data

class normalize_k_max(object):
    def __call__(self, data):
        input = data['fs_k_space']
        scaling_factor = input.abs().amax((1, 2, 3), keepdim=True)
        
        data['undersampled'] /= scaling_factor
        data['fs_k_space'] /= scaling_factor
        return data

class normalize_image_mean(object):
    def __call__(self, data):
        input = data['undersampled']
        img = root_sum_of_squares(ifft_2d_img(input), coil_dim=1)
        scaling_factor = img.mean((1, 2), keepdim=True).unsqueeze(1)

        data['undersampled'] /= scaling_factor
        data['fs_k_space'] /= scaling_factor
        return data

class normalize_image_mean2(object):
    def __call__(self, data):
        input = data['fs_k_space']
        img = root_sum_of_squares(ifft_2d_img(input), coil_dim=1)
        scaling_factor = 2*img.mean((1, 2), keepdim=True).unsqueeze(1)
        
        data['undersampled'] /= scaling_factor
        data['fs_k_space'] /= scaling_factor
        return data

class test_transform(object):
    def __call__(self, data):
        data, img = data

        k_space = data['undersampled']
        fs_k_space = data['fs_k_space'] 
        
        scaling_factor = k_space.abs().amax((1, 2, 3), keepdim=True)
        data['undersampled'] /= scaling_factor
        data['fs_k_space'] /= scaling_factor

        noisy_imgs = root_sum_of_squares(ifft_2d_img(data['fs_k_space']), coil_dim=1)
        noisy_imgs_max = noisy_imgs.amax((-1, -2), keepdim=True)
    
        img /= img.amax((-1, -2), keepdim=True)
        img *= noisy_imgs_max

        return data, img



