import os
import csv
import time
import h5py
from typing import Callable, Optional, Union, Collection
from argparse import ArgumentParser

from scipy.interpolate import interpn
import torch
import numpy as np

import nibabel as nib

from ml_recon.dataset.k_space_dataset import KSpaceDataset
from ml_recon.utils import fft_2d_img, ifft_2d_img, root_sum_of_squares

class KSpaceBrats(KSpaceDataset):
    """
    Takes data directory and creates a dataset. Before using you need to specify the file reader 
    to use in the filereader variable. 
    """

    def __init__(
            self,
            data_dir: Union[str, os.PathLike], 
            nx:int = 256,
            ny:int = 256,
            contrasts: Collection[str] = ['t1', 't2', 'flair', 't1ce'], 
            transforms: Optional[Callable] = None,
            extension: str = "npy"
            ):
        assert contrasts, 'Contrast list should not be empty!'

        super().__init__(nx=nx, ny=ny)

        self.transforms = transforms
        self.contrasts = np.array([contrast.lower() for contrast in contrasts])
        self.extension = extension

        sample_dir = os.listdir(data_dir)
        sample_dir.sort()

        slices = []
        data_list = []
        
        start = time.time()
        for sample in sample_dir:
            sample_path = os.path.join(data_dir, sample)
            sample_file = [file for file in os.listdir(sample_path) if 'h5' in file]
            sample_file_path = os.path.join(sample_path, sample_file[0])
            with h5py.File(sample_file_path, 'r') as fr:
                slices.append(fr['k_space'].shape[0])

            data_list.append(sample_file_path)

        self.file_list = np.array(data_list)
        _, self.contrast_order = self.get_data_from_indecies(0, 0)
        print(self.contrast_order)
        self.slices = np.array(slices)
        end = time.time()

        print(f'Elapsed time {(end-start)/60}')

        print(f'Found {sum(self.slices)} slices')

    # length of dataset is the sum of the slices
    def __len__(self):
        return sum(self.slices)

    def __getitem__(self, index):
        volume_index, slice_index = self.get_vol_slice_index(index)
        data, _ = self.get_data_from_indecies(volume_index, slice_index)

        if self.transforms:
            data = self.transforms(data)
        return data

    # get the volume index and slice index. This is done using the cumulative sum
    # of the number of slices.
    def get_vol_slice_index(self, index):
        cumulative_slice_sum = np.cumsum(self.slices)
        volume_index = np.sum(cumulative_slice_sum <= index)
        # if volume index is zero, slice is just index
        if volume_index == 0:
            slice_index = index
        # if volume index is larger than 1, its the cumulative sum of slices of volumes before subtracted
        # from the index
        else:
            slice_index = index - cumulative_slice_sum[volume_index - 1] 
        
        return volume_index, slice_index 
    
    def get_data_from_indecies(self, volume_index, slice_index):
        file = self.file_list[volume_index]
        with h5py.File(file, 'r') as fr:
            contrasts = fr['contrasts'][:].astype('U')

            use_modality_index = np.isin(contrasts, self.contrasts)
            modality_label = contrasts[use_modality_index]

            data = fr['k_space'][slice_index, use_modality_index, :, :, :]
            
        return torch.as_tensor(data), modality_label

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = KSpaceDataset.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False)

        parser.add_argument(
                '--data_dir', 
                type=str, 
                default='/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset/', 
                help=''
                )

        parser.add_argument(
                '--contrasts', 
                type=str, 
                nargs='+',
                default=['t1', 't2', 'flair', 't1ce'], 
                help=''
                )

        return parser

from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator
from ml_recon.transforms import normalize
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser = KSpaceBrats.add_model_specific_args(parser)
    args = parser.parse_args()
    dataset = KSpaceBrats(os.path.join(args.data_dir, 'train'), contrasts=args.contrasts, extension='nii.gz')
    dataset = UndersampleDecorator(dataset, transforms=normalize())

    counter = 0
    for i in dataset:
        x = i[0]
        counter += 1
        if counter > 1000: 
            break
