import numpy as np
import os
import json
from typing import Optional, Callable, Union
import torchvision.transforms.functional as F
import torch
import h5py
from argparse import ArgumentParser
from typing import List

from torch.utils.data import Dataset

class FastMRIDataset(Dataset):
    """This is a supervised slice dataloader. Returns data in [contrast, channel, height, width] where
    contrast dimension is 1

    Args:
        data_dir: Directory of h5py files
        nx: resolution in the x direction
        ny: resolution in the y direction
        build_new_header: builds a new header file so we don't have to every time
        
    """
    def __init__(
            self,
            data_dir: Union[str, os.PathLike],
            nx:int = 256,
            ny:int = 256,
            transforms: Optional[Callable] = None,
            contrasts: List[str] = ['t1']
            ):

        # call super constructor
        super().__init__()
        self.nx = nx
        self.ny = ny
        assert len(contrasts) == 1
        assert 't1' in contrasts

        self.transforms = transforms
        self.contrast_order = ['t1']

        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        files.sort()

        slices = []
        self.file_names = []
        for file in files:
            full_path = os.path.join(data_dir, file)
            with h5py.File(full_path) as fr:
                # loop through all the slices
                dataset = fr['kspace']
                assert dataset is h5py.Dataset
                slices.append(dataset.shape[0])
                self.file_names.append(full_path)
            
        self.slice_cumulative_sum = np.cumsum(slices)
        self.length = self.slice_cumulative_sum[-1]

        print(f'Found {self.slice_cumulative_sum[-1]} slices')


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        k_space = self.get_data_from_file(index)
        k_space = k_space.flip(1)
        k_space = self.resample_or_pad(k_space)

        # add contrast dimension
        k_space = k_space.unsqueeze(0)

        output = {
                'fs_k_space': k_space
                }

        if self.transforms:
            output = self.transforms(output)
        return output
    
    def get_data_from_file(self, index):
        volume_index = np.sum(self.slice_cumulative_sum <= index)
        slice_index = index if volume_index == 0 else index - self.slice_cumulative_sum[volume_index - 1]
        file_name = self.file_names[volume_index]
        with h5py.File(file_name) as fr:
            dataset = fr['kspace']
            assert dataset is h5py.Dataset
            k_space = torch.as_tensor(dataset[slice_index])

        return k_space 

    def resample_or_pad(self, k_space, reduce_fov=True):
        """Takes k-space data and resamples data to desired height and width. If 
        the image is larger, we crop. If the image is smaller, we pad with zeros

        Args:
            k_space (np.ndarray): k_space to be cropped or padded 
            reduce_fov (bool, optional): If we should reduce fov along readout dimension. Defaults to True.

        Returns:
            np.ndarray: cropped k_space
        """
        resample_height = self.ny
        resample_width = self.nx
        if reduce_fov:
            k_space = k_space[:, ::2, :]

        return F.center_crop(k_space, [resample_height, resample_width])

