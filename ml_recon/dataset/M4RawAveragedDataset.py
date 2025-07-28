import numpy as np
import os
from pathlib import Path
from typing import Union, Callable, List, Optional
import h5py

import torchvision.transforms.functional as F
import torch
from ml_recon.dataset.M4Raw_dataset import M4Raw
from torch.utils.data import Dataset

class M4RawAveraged(Dataset):
    """
    This is a dataloader for the M4Raw dataset. It loads a slice from the M4Raw 
    dataset without any subsampling.
    Attributes:
        nx (int): The desired width of the k-space data.
        ny (int): The desired height of the k-space data.
        key (str): The key to access the k-space data in the HDF5 files.
        transforms (Callable, optional): A function/transform to apply to the k-space data.
        contrast_order (np.ndarray): The order of contrasts in the dataset.
        contrast_order_indexes (np.ndarray): Boolean array indicating which contrasts are used.
        slice_cumulative_sum (np.ndarray): Cumulative sum of slices in the dataset.
        length (int): Total number of slices in the dataset.
        file_names (List[str]): List of file paths for the dataset.
    Methods:
        __len__(): Returns the total number of slices in the dataset.
        __getitem__(index): Returns the k-space data for the given index.
        get_data_from_file(index): Retrieves the k-space data from the file for the given index.
        center_k_space(contrast_k): Centers the k-space data for each contrast image.
        resample_or_pad(k_space): Resamples or pads the k-space data to the desired height and width.
    """

    def __init__(
            self,
            dataset, 
            test_dir: Union[str, Path],
            ):

        # call super constructor
        super().__init__()
        test_dir = Path(test_dir)
        self.dataset = dataset
        self.test_file_names = list(test_dir.iterdir())
        self.test_file_names.sort()

    def __len__(self):
        """
        Returns the total number of slices in the dataset.
        """
        return len(self.dataset)
    
    def __getitem__(self, index):
        k_space = self.dataset[index]
        vol_index, slice_index = self.dataset.dataset.get_file_indecies(index)
        cur_file = self.test_file_names[vol_index]

        
        
        with h5py.File(cur_file, 'r') as fr:
            dataset = fr['reconstruction_rss']
            assert isinstance(dataset, h5py.Dataset)
            image = dataset[self.dataset.dataset.contrast_order_indexes, slice_index]

        return k_space, torch.from_numpy(image)