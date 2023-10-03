import numpy as np
import os
import json
from typing import Union, Callable
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torch
import h5py
import random
from argparse import ArgumentParser

from ml_recon.utils.read_headers import make_header
from ml_recon.dataset.undersample import gen_pdf_columns, calc_k, apply_undersampling

class M4Raw(Dataset):
    """This is a dataloader for m4Raw. All it does is load a slice from the M4Raw 
    dataset. It does not do any subsampling

    Args:
    """
    def __init__(
            self,
            data_dir: Union[str, os.PathLike],
            nx:int = 256,
            ny:int = 256,
            transforms: Union[Callable, None] = None, 
            build_new_header: bool = False
            ):

        # call super constructor
        super().__init__()

        self.transforms = transforms
        self.nx = nx
        self.ny = ny
        self.random_index = random.randint(0, 10000)

        data_list = []

        header_file = os.path.join(data_dir, 'header.json')

        if not os.path.isfile(header_file) or build_new_header:
            print(f'Making header in {data_dir}')
            data_list = make_header(data_dir, output=header_file)
        else:    
            with open(header_file, 'r') as f:
                print('Header file found!')
                index_info = json.load(f)
                for value in index_info:
                    data_list.append(value)
        
        slices = np.array([volume['slices'] for volume in data_list])
        self.file_names = np.array([volume['file_name'] for volume in data_list])
        self.slice_cumulative_sum = np.cumsum(slices)
        self.length = self.slice_cumulative_sum[-1]

        self.omega_prob = gen_pdf_columns(nx, ny, 1/R, poly_order, acs_lines)
        self.lambda_prob = gen_pdf_columns(nx, ny, 1/R_hat, poly_order, acs_lines)

        one_minus_eps = 1 - 1e-3
        self.lambda_prob[self.lambda_prob > one_minus_eps] = one_minus_eps

        self.k = torch.from_numpy(calc_k(self.lambda_prob, self.omega_prob)).float()
        
        print(f'Found {self.slice_cumulative_sum[-1]} slices')


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        k_space = self.get_data_from_file(index)
        k_space = k_space.flip(1)
        k_space = self.resample_or_pad(k_space)

        under = apply_undersampling(index + self.random_index, self.omega_prob, k_space, True)
        doub_under = apply_undersampling(index, self.lambda_prob, under, False)
        
        data = (doub_under, under, k_space, self.k)

        if self.transforms:
            data = self.transforms(data)
        return data
    
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

        return F.center_crop(k_space, (resample_height, resample_width))

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
                "--R", 
                default=4,
                type=int,
                help="Omega undersampling factor"
                )

        parser.add_argument(
                "--R_hat", 
                default=2,
                type=int,
                help="Lambda undersampling factor"
                )

        parser.add_argument(
                "--poly_order", 
                default=8,
                type=int,
                help="Polynomial order for undersampling"
                )

        parser.add_argument(
                "--nx", 
                default=256,
                type=int,
                help="Number of points in the x direction"
                )

        parser.add_argument(
                "--ny", 
                default=256,
                type=int,
                help="Number of points in the y direction"
                )

        parser.add_argument(
                "--acs_lines", 
                default=10,
                type=int,
                help="Number of lines to keep in auto calibration region"
                )

        parser.add_argument(
                '--data_dir', 
                type=str, 
                default='/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/', 
                help=''
                )

        return parser


if __name__ == '__main__':
    dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/multicoil_train/'
    dataset = SliceDataset(dir)
    l = dataset[0]
    t = dataset[100]
