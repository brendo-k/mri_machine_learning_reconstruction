# import standard lib modules
from typing import Optional, Union, Literal
from pathlib import Path
import os

# import DL modules
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# import my modules
from ml_recon.dataset.undersample_decorator import UndersampleDecorator
from ml_recon.utils import k_to_img
from ml_recon.dataset.BraTS_dataset import BratsDataset
from ml_recon.dataset.M4Raw_dataset import M4Raw
from ml_recon.dataset.FastMRI_dataset import FastMRIDataset
from ml_recon.dataset.test_dataset import TestDataset


ACS_LINES = int(os.getenv('ACS_LINES')) if os.getenv('ACS_LINES') else 10

class UndersampledDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        dataset_name: Literal['fastmri', 'm4raw', 'brats'],
        data_dir: str, 
        test_dir: str,
        batch_size: int, 
        R: float = 6,
        R_hat: float = 2.0,
        contrasts: list[str] = ['t1', 't1ce', 't2', 'flair'],
        resolution: tuple[int, int] = (128, 128),
        num_workers: int = 0,
        poly_order: int = 8,
        norm_method: Union[Literal['k', 'img', 'image_mean', 'image_mean2', 'std'], None] = 'image_mean',
        self_supervsied: bool = False,
        sampling_method: str = '2d',
        ssdu_partioning: bool = False,
        limit_volumes: Optional[Union[int, float]] = None,
        same_mask_every_epoch: bool = False,
    ):
        """
            dataset_name:

        """

        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.test_dir = Path(test_dir)
        self.contrasts = contrasts
        self.num_contrasts = len(contrasts)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.resolution = resolution
        self.R = R
        self.R_hat = R_hat
        self.poly_order = poly_order
        self.self_supervised = self_supervsied
        self.ssdu_partioning = ssdu_partioning
        self.sampling_method = sampling_method
        self.norm_method = norm_method
        self.limit_volumes = limit_volumes
        self.same_mask_every_epoch = same_mask_every_epoch

        self.dataset_class, self.test_data_key = self.setup_dataset_type(dataset_name)
        self.transforms = self.setup_data_normalization(norm_method)


    def setup(self, stage):
        super().setup(stage)


        # get directories for different split folders
        train_dir = self.data_dir / 'train'
        val_dir = self.data_dir / 'val'
        test_dir = self.data_dir / 'test'

        # ground truth denoised directories
        val_gt_dir = self.test_dir / 'val'
        test_gt_dir = self.test_dir / 'test'

        # keywords to control dataset data
        dataset_keyword_args = {
            'nx': self.resolution[0], 
            'ny': self.resolution[1],
            'contrasts': self.contrasts, 
            'limit_volumes': self.limit_volumes
        }

        # keywords to control k-space undersamplig 
        undersample_keyword_args = {
            'R': self.R,
            'R_hat': self.R_hat,
            'sampling_method': self.sampling_method,
            'self_supervised': self.self_supervised,
            'acs_lines' : ACS_LINES, 
            'poly_order': self.poly_order,
            'original_ssdu_partioning': self.ssdu_partioning,
            'same_mask_every_epoch': self.same_mask_every_epoch
        }

        # undersampled training dataset
        self.train_dataset = UndersampleDecorator(
            self.dataset_class(
                train_dir, 
                **dataset_keyword_args
            ),
            transforms=self.transforms,
            **undersample_keyword_args
        )


        # undersampled validation dataset
        noisy_val_dataset_undersampled = UndersampleDecorator(
            self.dataset_class(
                val_dir, 
                **dataset_keyword_args
            ),
            transforms=self.transforms,
            **undersample_keyword_args
        )
        # denoised val dataset
        gt_val_dataset = self.dataset_class(
            val_gt_dir, 
            data_key=self.test_data_key, 
            **dataset_keyword_args
        )
        # both noisy and ground truth val dataset
        self.val_dataset = TestDataset(
            noisy_val_dataset_undersampled,
            gt_val_dataset, 
            transforms=None
        )

        # noisy test dataset
        noisy_test_dataset_undersampled = UndersampleDecorator(
            self.dataset_class(
                test_dir, 
                **dataset_keyword_args
            ),
            **undersample_keyword_args,
            transforms=self.transforms
        )
        # ground truth test dataset
        gt_test_dataset = self.dataset_class(
            test_gt_dir, 
            data_key=self.test_data_key, 
            **dataset_keyword_args
        )
        # both datasets combined
        self.test_dataset = TestDataset(
            noisy_test_dataset_undersampled,
            gt_test_dataset, 
            transforms=None
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
    def setup_dataset_type(self, dataset_name):
        dataset_name = str.lower(dataset_name)
        if dataset_name == 'brats': 
            dataset_class = BratsDataset
            test_data_key = 'ground_truth'
        elif dataset_name == 'fastmri':
            dataset_class = FastMRIDataset
            test_data_key = 'reconstruction_rss'
        elif dataset_name == 'm4raw':
            dataset_class = M4Raw
            test_data_key = 'reconstruction_rss'
        else: 
            raise ValueError(f'{dataset_name} is not a valid dataset name')
        return dataset_class, test_data_key

    def setup_data_normalization(self, norm_method):
        if norm_method == 'img':
            transforms = normalize_image_max()
        elif norm_method == 'k': 
            transforms = normalize_k_max() 
        elif norm_method == 'norm_l2':
            transforms = normalize_l2() 
        elif norm_method == 'image_mean':
            transforms = normalize_image_mean() 
        elif norm_method == 'std':
            transforms = normalize_image_std()
        else:
            transforms = None
        return transforms

class normalize_image_max(object):
    def __call__(self, data: dict):
        input = data['fs_k_space']
        img = k_to_img(input, coil_dim=1)
        scaling_factor = img.amax((1, 2), keepdim=True).unsqueeze(1)

        data['undersampled'] /= scaling_factor
        data['fs_k_space'] /= scaling_factor
        data['scaling_factor'] = scaling_factor
        return data

class normalize_k_max(object):
    def __call__(self, data):
        input = data['undersampled']
        scaling_factor = input.abs().amax((1, 2, 3), keepdim=True)

        data['undersampled'] /= scaling_factor
        data['fs_k_space'] /= scaling_factor
        data['scaling_factor'] = scaling_factor
        return data

class normalize_l2(object):
    def __call__(self, data):
        input = data['fs_k_space']
        img = k_to_img(input, coil_dim=1)
        scaling_factor = img.norm(2, (1, 2), keepdim=True).unsqueeze(1)

        data['undersampled'] /= scaling_factor
        data['scaling_factor'] = scaling_factor
        data['fs_k_space'] /= scaling_factor
        return data

class normalize_image_mean(object):
    def __call__(self, data):
        input = data['undersampled']
        img = k_to_img(input, coil_dim=1)
        scaling_factor = img.mean((1, 2), keepdim=True).unsqueeze(1)

        data['undersampled'] /= scaling_factor
        data['scaling_factor'] = scaling_factor
        data['fs_k_space'] /= scaling_factor
        return data

class normalize_image_std(object):
    def __call__(self, data):
        input = data['undersampled']
        img = k_to_img(input, coil_dim=1)
        scaling_factor = img.std((1, 2), keepdim=True).unsqueeze(1)

        data['undersampled'] /= scaling_factor
        data['scaling_factor'] = scaling_factor
        data['fs_k_space'] /= scaling_factor
        return data

