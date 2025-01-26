import numpy as np
import os
from typing import Union, Callable, List
import torchvision.transforms.functional as F
import torch
import h5py

from torch.utils.data import Dataset
from typing import Union, Optional

class M4Raw(Dataset):
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
            data_dir: Union[str, os.PathLike],
            nx:int = 256,
            ny:int = 256,
            transforms: Union[Callable, None] = None, 
            key:str = 'kspace',
            contrasts: List[str] = ['t1', 't2', 'flair'], 
            limit_volumes: Optional[Union[int, float]] = None
            ):

        # call super constructor
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.key = key

        self.transforms = transforms

        files = os.listdir(data_dir)
        self.file_names = []
        slices = []
        contrast_order = []

        files.sort()

        if limit_volumes is None:
            limit_volumes = len(files)
        elif isinstance(limit_volumes, float):
            limit_volumes = int(limit_volumes * len(files))
            
        for file in files[:limit_volumes]:
            file_path = os.path.join(data_dir, file)
            self.file_names.append(file_path)

            with h5py.File(file_path, 'r') as fr:
                dataset = fr[self.key]
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
            dataset = fr[self.key]
            assert isinstance(dataset, h5py.Dataset)
            k_space = dataset[self.contrast_order_indexes, slice_index]
            
            # extra processing for k-space data
            if self.key == 'kspace':
                k_space = self.center_k_space(k_space)
                k_space = M4Raw.fill_missing_k_space(k_space)
                
        return k_space 

    def center_k_space(self, contrast_k):
        """
        Centers the k-space data for each contrast image in the input array.

        This function finds the maximum value in each 2D k-space slice of the input
        array `contrast_k` and shifts the k-space data such that the maximum value
        is centered. The centering is done by rolling the array along the last two
        axes.

        Parameters:
        contrast_k (numpy.ndarray): A 3D numpy array of k-space data with shape
                                    (num_contrasts, height, width).

        Returns:
        numpy.ndarray: The centered k-space data with the same shape as the input.
        """
        for i in range(contrast_k.shape[0]):
            _, h_center, w_center = np.unravel_index(np.argmax(contrast_k[i]), contrast_k[i].shape)
            diff_h = h_center - contrast_k[i].shape[-2]//2
            diff_w = w_center - contrast_k[i].shape[-1]//2
            contrast_k[i] = np.roll(contrast_k[i], (-diff_h, -diff_w), axis=(-2, -1))
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
    
    @staticmethod
    def fill_missing_k_space(k_space):
        # if there is missing data on one of the coils, we replace it with the average of the other coils
        
        contrast, coils, h, w = k_space.shape
        zero_fill_mask = np.zeros_like(k_space) 
        zero_fill_mask[:, :, :, 31:-30] = 1

        zeros_mask = k_space == 0

        # Compute the indices of the maximum absolute values along the channel (c) dimension
        max_indices = np.argmax(np.abs(k_space), axis=1, keepdims=True)  # Shape: (b, 1, h, w)

        # Use take_along_axis to gather the maximum values while retaining the complex data
        averaged_k = np.take_along_axis(k_space, max_indices, axis=1)

        averaged_k = np.tile(averaged_k, (1, coils, 1, 1))

        k_space[zeros_mask] = averaged_k[zeros_mask]

        return k_space

