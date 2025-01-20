import os
import csv
import time
import h5py
from typing import Callable, Optional, Union, Collection
import torchvision.transforms.functional as F 
from ml_recon.dataset.BraTS_dataset import BratsDataset
from ml_recon.dataset.undersample_decorator import UndersampleDecorator

import torch
import numpy as np


from torch.utils.data import Dataset

class BratsDatasetTest(Dataset):
    """
    Takes data directory and creates a dataset for BraTS dataset. Need to simulate first 
    using ml_recon/utils/simulate_brats
    """

    def __init__(
            self,
            data_dir: str, 
            ground_truth_dir: Union[str, os.PathLike], 
            nx:int = 256,
            ny:int = 256,
            contrasts: Collection[str] = ['t1', 't2', 'flair', 't1ce'], 
            transforms: Optional[Callable] = None,
            R: int=4,
            R_hat: int=2,
            acs_lines: int=10,
            sampling_method:str='2d',
            self_supervised:bool=False
            ):

        assert contrasts, 'Contrast list should not be empty!'
        super().__init__()

        self.transforms = transforms
        self.undersampled_dataset = UndersampleDecorator(
                BratsDataset(data_dir, nx, ny, contrasts),
                R_hat=R_hat,
                R=R,
                sampling_method=sampling_method,
                self_supervised=self_supervised, 
                acs_lines=acs_lines
                )

        self.ground_truth_dataset = BratsDataset(ground_truth_dir, nx, ny, contrasts, data_key='rss_images')

    def __len__(self):
        return len(self.undersampled_dataset)

    def __getitem__(self, index):
        k_space = self.undersampled_dataset[index]
        ground_truth = torch.from_numpy(self.ground_truth_dataset[index])

        if self.transforms:
            k_space, ground_truth = self.transforms((k_space, ground_truth))

        return k_space, ground_truth
        
