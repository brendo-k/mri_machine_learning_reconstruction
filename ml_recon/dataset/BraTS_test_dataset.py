import os
import csv
import time
import h5py
from typing import Callable, Optional, Union, Collection
import torchvision.transforms.functional as F 

import torch
import numpy as np


from torch.utils.data import Dataset

class BratsDataset(Dataset):
    """
    Takes data directory and creates a dataset for BraTS dataset. Need to simulate first 
    using ml_recon/utils/simulate_brats
    """

    def __init__(
            self,
            data_dir: Union[str, os.PathLike], 
            ground_truth_dir: Union[str, os.PathLike], 
            nx:int = 256,
            ny:int = 256,
            contrasts: Collection[str] = ['t1', 't2', 'flair', 't1ce'], 
            transforms: Optional[Callable] = None,
            extension: str = "npy"
            ):
        assert contrasts, 'Contrast list should not be empty!'

        super().__init__()
        self.nx = nx
        self.ny = ny
        self.transforms = transforms
        self.contrasts = np.array([contrast.lower() for contrast in contrasts])
        self.extension = extension

        sample_dir = os.listdir(data_dir)
        sample_dir.sort()

        slices = []
        data_list = []
        image_list = []
        contrast_order = []
        
        start = time.time()
        first = True
        for sample in sample_dir:
            sample_path = os.path.join(data_dir, sample)
            sample_file = [file for file in os.listdir(sample_path) if 'h5' in file]
            sample_file_path = os.path.join(sample_path, sample_file[0])
            with h5py.File(sample_file_path, 'r') as fr:
                k_space = fr['k_space']
                assert isinstance(k_space, h5py.Dataset)
                num_slices = k_space.shape[0]
                slices.append(num_slices)
                if first:
                    contrast_dataset = fr['contrasts']
                    assert isinstance(contrast_dataset, h5py.Dataset)
                    contrast_order = contrast_dataset[:].astype('U')
                    first = False

            data_list.append(sample_file_path)

            sample_path = os.path.join(ground_truth_dir, sample)
            gt_file = [file for file in os.listdir(sample_path) if 'h5' in file]
            image_list.append(gt_file)


        end = time.time()
        print(f'Elapsed time {(end-start)/60}')


        self.contrast_order_indexes = np.isin(contrast_order, self.contrasts)
        self.contrast_order = contrast_order[self.contrast_order_indexes] # type: ignore
        
        self.file_list = np.array(data_list)
        self.image_list = np.array(image_list)
        print(self.contrast_order)
        self.slices = np.array(slices)
        self.cumulative_slice_sum = np.cumsum(self.slices)
        self.length = self.cumulative_slice_sum[-1]

        print(f'Found {sum(self.slices)} slices')

    # length of dataset is the sum of the slices
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        volume_index, slice_index = self.get_vol_slice_index(index)
        k_space, imgs = self.get_data_from_indecies(volume_index, slice_index)

        if self.transforms:
            data = self.transforms(data)

        return data

    # get the volume index and slice index. This is done using the cumulative sum
    # of the number of slices.
    def get_vol_slice_index(self, index):
        volume_index = np.sum(self.cumulative_slice_sum <= index)
        # if volume index is zero, slice is just index
        if volume_index == 0:
            slice_index = index
        # if volume index is larger than 1, its the cumulative sum of slices of volumes before subtracted
        # from the index
        else:
            slice_index = index - self.cumulative_slice_sum[volume_index - 1] 
        
        return volume_index, slice_index 
    
    def get_data_from_indecies(self, volume_index, slice_index):
        file = self.file_list[volume_index]
        img_file = self.image_list[volume_index]

        with h5py.File(file, 'r') as fr:
            dataset = fr['k_space']
            assert isinstance(dataset, h5py.Dataset)
            data = torch.as_tensor(dataset[slice_index, self.contrast_order_indexes])
            data = F.center_crop(data, [self.ny, self.nx]).numpy()

        with h5py.File(img_file, 'r') as fr:
            imgs = fr['rss_image'] 
            assert isinstance(imgs, h5py.Dataset)
            imgs = imgs[slice_index, self.contrast_order_indexes]

        return data, imgs


