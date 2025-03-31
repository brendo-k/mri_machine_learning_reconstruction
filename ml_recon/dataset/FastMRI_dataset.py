import numpy as np
import os
import json
from typing import Optional, Callable, Union, List
import torchvision.transforms.functional as F
import torch
import h5py
from ml_recon.utils.image_processing import k_to_img  

from torch.utils.data import Dataset

class FastMRIDataset(Dataset):
    """This is a slice dataloader. Returns data in [contrast, channel, height, width] where
    contrast dimension is 1

    """
    def __init__(
            self,
            data_dir: Union[str, os.PathLike],
            nx:int = 256,
            ny:int = 256,
            transforms: Optional[Callable] = None,
            contrasts: List[str] = ['t1'], 
            data_key = 'kspace',
            limit_volumes: Optional[Union[int, float]] = None
            ):

        # call super constructor
        super().__init__()
        self.nx = nx
        self.ny = ny
        assert len(contrasts) == 1
        assert 't1' in contrasts

        self.transforms = transforms
        self.contrast_order = ['t1']
        self.key = data_key

        sample_dir = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        sample_dir.sort()

        slices = []
        self.file_names = []
        if limit_volumes is None:
            limit_volumes = len(sample_dir)
        elif isinstance(limit_volumes, float):
            limit_volumes = int(limit_volumes * len(sample_dir))
        elif isinstance(limit_volumes, int):
            limit_volumes = limit_volumes
            
        for sample in sample_dir[:limit_volumes]:
            full_path = os.path.join(data_dir, sample)
            with h5py.File(full_path, 'r') as fr:
                # loop through all the slices
                dataset = fr[self.key]
                assert isinstance(dataset, h5py.Dataset)
                slices.append(dataset.shape[0])
                self.file_names.append(full_path)
            
        self.slice_cumulative_sum = np.cumsum(slices)
        self.length = self.slice_cumulative_sum[-1]

        print(f'Found {self.slice_cumulative_sum[-1]} slices')


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.get_data_from_file(index)
        data = data.flip(1)
        if self.key == 'kspace':
            data = self.resample_or_pad(data)
        else: 
            data = self.resample_or_pad(data)
            imgs = k_to_img(k_space=data, coil_dim=0)
            data = imgs
            

        # add contrast dimension
        data = data.unsqueeze(0).numpy()

        if self.transforms:
            data = self.transforms(data)
        return data
    
    def get_data_from_file(self, index):
        volume_index = np.sum(self.slice_cumulative_sum <= index)
        slice_index = index if volume_index == 0 else index - self.slice_cumulative_sum[volume_index - 1]
        file_name = self.file_names[volume_index]
        with h5py.File(file_name, 'r') as fr:
            dataset = fr['kspace']
            assert isinstance(dataset, h5py.Dataset)
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

