import numpy as np
import os
from typing import Union, Callable, List
import torchvision.transforms.functional as F
import torch
import h5py

from torch.utils.data import Dataset

class M4Raw(Dataset):
    """This is a dataloader for m4Raw. All it does is load a slice from the M4Raw 
    dataset. It does not do any subsampling

    """
    def __init__(
            self,
            data_dir: Union[str, os.PathLike],
            nx:int = 256,
            ny:int = 256,
            transforms: Union[Callable, None] = None, 
            contrasts: List[str] = ['t1', 't2', 'flair']
            ):

        # call super constructor
        super().__init__()
        self.nx = nx
        self.ny = ny

        self.transforms = transforms

        files = os.listdir(data_dir)
        self.file_names = []
        slices = []
        contrast_order = []

        files.sort()
        for file in files:
            file_path = os.path.join(data_dir, file)
            self.file_names.append(file_path)

            with h5py.File(file_path, 'r') as fr:
                dataset = fr['kspace']
                assert isinstance(dataset, h5py.Dataset)
                slices.append(dataset.shape[1])
                contrast_dataset = fr['contrasts']
                assert isinstance(contrast_dataset, h5py.Dataset)
                contrast_order = np.char.lower(contrast_dataset[:].astype('U'))

        contrasts_arr = np.array(contrasts)
        self.contrast_order_indexes = np.isin(contrast_order, contrasts_arr)
        self.contrast_order = contrast_order[self.contrast_order_indexes] # type: ignore

        self.slice_cumulative_sum = np.cumsum(slices) 
        self.length = self.slice_cumulative_sum[-1]
        print(f'Found {self.length} slices!')
        print(f'Found {self.contrast_order} contrats!!')


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
        cur_file = self.file_names[volume_index]
        
        
        with h5py.File(cur_file, 'r') as fr:
            dataset = fr['kspace']
            assert isinstance(dataset, h5py.Dataset)
            k_space = dataset[self.contrast_order_indexes, slice_index]
                
        k_space = np.stack([self.center_k_space(contrast) for contrast in k_space], axis=0)
        return k_space 

    def center_k_space(self, contrast_k):
        _, h_center, w_center = np.unravel_index(np.argmax(contrast_k), contrast_k.shape)
        diff_h = h_center - contrast_k.shape[-2]//2
        diff_w = w_center - contrast_k.shape[-1]//2
        contrast_k = np.roll(contrast_k, (-diff_h, -diff_w), axis=(-2, -1))
        return contrast_k

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

        return F.center_crop(torch.from_numpy(k_space), [resample_height, resample_width]).numpy()
