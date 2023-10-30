import numpy as np
import os
from typing import Union, Callable, Collection
import torchvision.transforms.functional as F
import torch
import h5py
from argparse import ArgumentParser

from ml_recon.dataset.k_space_dataset import KSpaceDataset

class M4Raw(KSpaceDataset):
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
            contrasts: Collection[str] = ['t1', 't2', 'flair']
            ):

        # call super constructor
        super().__init__(nx=nx, ny=ny)

        self.transforms = transforms

        files = os.listdir(data_dir)
        patient_id = list(set([file.split('-')[0] for file in files]))
        patient_id.sort()
        
        slices = []
        self.file_names = []

        first = True
        for patient in patient_id:
            patient_files = [os.path.join(data_dir, file) for file in files if patient in file]
            patient_files.sort()

            if first: 
                self.contrast_order = np.array([file.split('-')[-1].split('.')[0] for file in patient_files])
                first = False
            
            self.file_names.append(np.array(patient_files))

            with h5py.File(patient_files[0], 'r') as fr:
                slices.append(fr['kspace'].shape[0])


        contrasts = np.array(contrasts)
        print(self.contrast_order)
        self.contrast_index = np.isin(self.contrast_order, contrasts)
        print(self.contrast_index)
        self.contrasts = self.contrast_order[self.contrast_index]

        self.slice_cumulative_sum = np.cumsum(slices) 
        self.length = self.slice_cumulative_sum[-1]
        print(f'Found {self.length} slices!')
        print(f'Found {self.contrasts} slices!')


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        k_space = self.get_data_from_file(index)
        k_space = self.resample_or_pad(k_space)

        if self.transforms:
            k_space = self.transforms(k_space)
        return k_space
    
    def get_data_from_file(self, index):
        volume_index = np.sum(self.slice_cumulative_sum <= index)
        slice_index = index if volume_index == 0 else index - self.slice_cumulative_sum[volume_index - 1]
        cur_files = self.file_names[volume_index]
        print(cur_files)
        cur_files = cur_files[self.contrast_index]
        
        k_space = []
        
        for file in cur_files:
            with h5py.File(file) as fr:
                k_space.append(torch.as_tensor(fr['kspace'][slice_index]))

        k_space = torch.stack(k_space, axis=0)
        return k_space 

    def resample_or_pad(self, k_space):
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

        return F.center_crop(k_space, (resample_height, resample_width))

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
                '--data_dir', 
                type=str, 
                default='/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/', 
                help=''
                )

        return parser


import matplotlib.pyplot as plt 
from ml_recon.utils import image_slices, root_sum_of_squares, ifft_2d_img
if __name__ == '__main__':
    dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/M4raw/multicoil_train_averaged/'
    dataset = M4Raw(dir, contrasts=['t1', 't2'])
    l = dataset[0]

    image_slices(l[0].abs(), cmap='gray')
    image_slices(ifft_2d_img(l[0]).abs(), cmap='gray')
    plt.figure(3)
    plt.imshow(root_sum_of_squares(ifft_2d_img(l[0]), coil_dim=0), cmap='gray')
    plt.figure(4)
    plt.imshow(root_sum_of_squares(ifft_2d_img(l[1]), coil_dim=0), cmap='gray')
    plt.show()


