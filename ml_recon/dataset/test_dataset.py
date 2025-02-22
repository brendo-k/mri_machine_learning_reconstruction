import numpy as np
import os
from typing import Union, Callable, List, Optional
import torchvision.transforms.functional as F
from ml_recon.dataset.M4Raw_dataset import M4Raw
from ml_recon.dataset.undersample_decorator import UndersampleDecorator
import torch

from torch.utils.data import Dataset

class TestDataset(Dataset):
    """This is a dataset the composes two datasets. One ground truth or denoised dataset
    and the original undersampled dataset. 
    Called in pl_undersampledDataModule 

    """
    def __init__(
            self,
            undersampled_dataset, 
            ground_truth_dataset,
            transforms
            ):

        # call super constructor
        super().__init__()
        self.undersampled_dataset = undersampled_dataset
        self.ground_truth_dataset = ground_truth_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.undersampled_dataset)

    def __getitem__(self, index):
        k_space = self.undersampled_dataset[index]
        image = torch.from_numpy(self.ground_truth_dataset[index])
        if self.transforms: 
            k_space, image = self.transforms((k_space, image))
        return k_space, image

