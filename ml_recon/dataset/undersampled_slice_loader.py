from .slice_loader import SliceLoader 
import random
import numpy as np
from .FileReader.read_h5 import H5FileReader
import os
from typing import Union, Callable

class UndersampledSliceDataset(SliceLoader):
    def __init__(
            self, 
            meta_data: Union[str, os.PathLike],
            acs_width: int=20,
            R: int=8, 
            raw_sample_filter: Callable=lambda _: True,
            transforms: Callable=None):

        # call super constructor 
        super().__init__(meta_data, raw_sample_filter=raw_sample_filter)
        # define our file reader
        super().set_file_reader(H5FileReader)
        self.acs_width = acs_width
        # add transforms
        self.transforms = transforms
        self.R = R

    def __getitem__(self, index):
        data = super().__getitem__(index)

        # get k-space data
        k_space = data['k_space']

        random_indecies = self.get_undersampled_indecies(k_space, self.acs_width, self.R)
        undersampled = self.apply_undersampled_indecies(k_space, random_indecies)
        mask = self.build_mask(k_space, random_indecies)

        data['mask'] = mask
        data['undersampled'] = undersampled

        if self.transforms:
            data = self.transforms(data)
        return data

    def build_mask(self, k_space, random_indecies) :
        k_space_size = self.get_k_space_size(k_space)
        mask = np.ones((k_space_size[0], k_space_size[1]), dtype=np.int8)
        mask[..., random_indecies] = 0
        mask = mask.astype(bool)
        return mask

    def apply_undersampled_indecies(self, k_space, random_indecies):
        undersampled = np.copy(k_space)
        undersampled[..., random_indecies] = 0

        return undersampled

    def get_undersampled_indecies(self, k_space, acs_width, R):
        k_space_size = self.get_k_space_size(k_space)
        center = int(k_space_size[1]//2)
        acs_bounds = [(center - acs_width//2), (center + np.ceil(acs_width//2).astype(int))]
        sampled_indeces = [index for index in range(k_space_size[1]) if index not in range(acs_bounds[0],acs_bounds[1])]
        random_indecies = random.choices(sampled_indeces, k=k_space_size[1]//R)
        random_indecies = np.concatenate((random_indecies, range(acs_bounds[0], acs_bounds[1])))

        all_indexes = np.arange(0, k_space_size[1])
        undersampled_indeces = np.setdiff1d(all_indexes, random_indecies) 
        return undersampled_indeces

    def get_k_space_size(self, k_space):
        return k_space.shape[-2:] 