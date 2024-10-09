import numpy as np
import os
from typing import Union, Callable, Collection
import torchvision.transforms.functional as F
import torch
import h5py
from argparse import ArgumentParser

from torch.utils.data import Dataset

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
            contrasts: Collection[str] = ['t1', 't2', 'flair']
            ):

        # call super constructor
        super().__init__()
        self.nx = nx
        self.ny = ny

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
                dataset = fr['kspace']
                assert dataset is h5py.Dataset
                slices.append(dataset.shape[0])


        contrasts = np.array(contrasts)
        self.contrast_index = np.isin(self.contrast_order, contrasts)
        self.contrast_order = self.contrast_order[self.contrast_index]

        self.slice_cumulative_sum = np.cumsum(slices) 
        self.length = self.slice_cumulative_sum[-1]
        print(f'Found {self.length} slices!')
        print(f'Found {self.contrast_order} contrats!!')


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        k_space = self.get_data_from_file(index)
        k_space = self.resample_or_pad(k_space)

        output = {
                'fs_k_space': k_space
                }

        if self.transforms:
            output = self.transforms(output)

        return output
    
    def get_data_from_file(self, index):
        volume_index = np.sum(self.slice_cumulative_sum <= index)
        slice_index = index if volume_index == 0 else index - self.slice_cumulative_sum[volume_index - 1]
        cur_files = self.file_names[volume_index]
        cur_files = cur_files[self.contrast_index]
        
        k_space = []
        
        for file in cur_files:
            with h5py.File(file) as fr:
                dataset = fr['kspace']
                assert dataset is h5py.Dataset
                k_space.append(torch.as_tensor(dataset[slice_index]))

        k_space = torch.stack(k_space, dim=0)
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

        return F.center_crop(k_space, [resample_height, resample_width])
