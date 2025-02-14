import numpy as np
import os
from typing import Union, Callable, List, Optional
import torchvision.transforms.functional as F
from ml_recon.dataset.M4Raw_dataset import M4Raw
from ml_recon.dataset.undersample_decorator import UndersampleDecorator
import torch
import h5py

from torch.utils.data import Dataset

class M4RawTest(Dataset):
    """This is a test dataloader for m4Raw. It loads one k-space slice from the first 
    average and an image slice from the averaged data

    """
    def __init__(
            self,
            data_dir: Union[str, os.PathLike],
            test_dir: Union[str, os.PathLike],
            nx:int = 256,
            ny:int = 256,
            transforms: Union[Callable, None] = None, 
            contrasts: List[str] = ['t1', 't2', 'flair'],
            R: int = 4, 
            R_hat: int = 2,
            self_supervised: bool = True, 
            sampling_method: str = '2d',
            acs_lines: int = 10, 
            limit_volumes: Optional[Union[int, float]] = None
            ):

        # call super constructor
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.transforms = transforms

        self.k_space_dataset = UndersampleDecorator(
                dataset=M4Raw(data_dir, nx, ny, contrasts=contrasts, limit_volumes=limit_volumes),
                R=R, 
                R_hat=R_hat, 
                sampling_method = sampling_method,
                self_supervised = self_supervised, 
                acs_lines=acs_lines
                )
        self.average_dataset = M4Raw(test_dir, nx, ny, contrasts=contrasts, key='reconstruction_rss', limit_volumes=limit_volumes)

    def __len__(self):
        return len(self.k_space_dataset)

    def __getitem__(self, index):
        k_space = self.k_space_dataset[index]
        image = torch.from_numpy(self.average_dataset[index])
        scaling_factor = 1
        if self.transforms: 
            k_space, image, scaling_factor = self.transforms((k_space, image))
        return k_space, image, scaling_factor

