from .kspace_dataset import KSpaceDataset
import random
import numpy as np
from .FileReader.read_h5 import H5FileReader

class UndersampledKSpaceDataset(KSpaceDataset):
    def __init__(self, h5_directory, transforms=None):
        # call super constructor 
        super().__init__(h5_directory)
        # define our file reader
        self.filereader = H5FileReader

        # add transforms
        self.transforms = transforms

    def __getitem__(self, index):
        data = super().__getitem__(index)

        # get k-space data
        k_space = data['k_space']
        k_space_size = k_space.shape[-2:]

        random_indecies = self.get_undersampled_indecies(k_space_size)
        undersampled = self.apply_undersampled_indecies(k_space, random_indecies)
        mask = self.build_mask(k_space, k_space_size, random_indecies)

        data['k_space'] = k_space
        data['mask'] = mask
        data['undersampled'] = undersampled

        if self.transforms:
            data = self.transforms(data)
        return data

    def build_mask(self, k_space, k_space_size, random_indecies):
        mask = np.ones((k_space.shape[0], k_space_size[0], k_space_size[1]), dtype=np.int8)
        mask[:, :, random_indecies] = 0
        return mask

    def apply_undersampled_indecies(self, k_space, random_indecies):
        undersampled = np.copy(k_space)
        undersampled[:, :, :, random_indecies] = 0
        return undersampled

    def get_undersampled_indecies(self, k_space_size):
        random_indecies = random.choices(range(k_space_size[1]), k=k_space_size[1]//2)
        return random_indecies
        