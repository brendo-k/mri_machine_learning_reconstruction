import numpy as np
import os
import json
from typing import Optional, Callable, Union
import torchvision.transforms.functional as F
import torch
import h5py
from argparse import ArgumentParser

from ml_recon.utils.read_headers import make_header
from ml_recon.dataset.k_space_dataset import KSpaceDataset

class FastMRIDataset(KSpaceDataset):
    """This is a supervised slice dataloader. 

    Args:
        data_dir: Directory of h5py files
        nx: resolution in the x direction
        ny: resolution in the y direction
        raw_sample_filter: how to sample slices
    """
    def __init__(
            self,
            data_dir: Union[str, os.PathLike],
            nx:int = 256,
            ny:int = 256,
            build_new_header: bool = False,
            transforms: Optional[Callable] = None,
            undersample: Optional[Callable] = None
            ):

        # call super constructor
        super().__init__(nx=nx, ny=ny)

        self.transforms = transforms
        self.undersample = undersample

        header_file = os.path.join(data_dir, 'header.json')
        if not os.path.isfile(header_file) or build_new_header:
            print(f'Making header in {data_dir}')
            volume_list = make_header(data_dir, output=header_file)
        else:    
            with open(header_file, 'r') as f:
                print('Header file found!')
                volume_list = json.load(f)
        
        slices = np.array([volume['slices'] for volume in volume_list])
        self.file_names = np.array([volume['file_name'] for volume in volume_list])
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

        if self.undersample:
            k_space = self.undersample(k_space)

        if self.transforms:
            k_space = self.transforms(k_space)
        return k_space
    
    def get_data_from_file(self, index):
        volume_index = np.sum(self.slice_cumulative_sum <= index)
        slice_index = index if volume_index == 0 else index - self.slice_cumulative_sum[volume_index - 1]
        file_name = self.file_names[volume_index]
        with h5py.File(file_name) as fr:
            k_space = torch.as_tensor(fr['kspace'][slice_index])
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

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = KSpaceDataset.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False)

        parser.add_argument(
                '--data_dir', 
                type=str, 
                default='/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/16_chans/' ,
                help='Top data directory where multicoil_train, multicoil_val, multicoil_test directories are'
                )

        return parser
