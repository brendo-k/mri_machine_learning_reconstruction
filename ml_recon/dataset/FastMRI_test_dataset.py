import numpy as np
import os
import json
from typing import Optional, Callable, Union
import torchvision.transforms.functional as F
from ml_recon.dataset.undersample_decorator import UndersampleDecorator
from ml_recon.dataset.FastMRI_dataset import FastMRIDataset
from ml_recon.utils import k_to_img
import torch
import h5py
from typing import List

from torch.utils.data import Dataset

class FastMRIDatasetTest(Dataset):
    """This is a slice dataloader. Returns data in [contrast, channel, height, width] where
    contrast dimension is 1

    """
    def __init__(
            self,
            data_dir: Union[str, os.PathLike],
            test_dir: Union[str, os.PathLike],
            nx:int = 256,
            ny:int = 256,
            transforms: Optional[Callable] = None,
            contrasts: List[str] = ['t1'], 
            R_hat: float = 4,
            R: float = 4,
            sampling_method: str = '2d',
            self_supervised: bool = False
            ):

        super().__init__()

        self.transforms = transforms
        self.undersampled_dataset = UndersampleDecorator(
                FastMRIDataset(data_dir, nx, ny, None, contrasts),
                R_hat=R_hat,
                R=R,
                sampling_method=sampling_method,
                self_supervised=self_supervised, 
                )

        self.ground_truth_dataset = FastMRIDataset(data_dir, nx, ny, None, contrasts, key='kspace')


    def __len__(self):
        return len(self.undersampled_dataset)

    def __getitem__(self, index):
        k_space = self.undersampled_dataset[index]
        ground_truth = torch.from_numpy(self.ground_truth_dataset[index])
        print(ground_truth.shape)
        ground_truth = k_to_img(ground_truth, coil_dim=1)

        if self.transforms:
            k_space, ground_truth = self.transforms((k_space, ground_truth))

        return k_space, ground_truth
