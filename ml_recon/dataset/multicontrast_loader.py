import os
from typing import (
    Callable,
    Optional,
    Union,
)
from torch.utils.data import Dataset
import torch
#from scipy.interpolate import interpn
import numpy as np

from ml_recon.dataset.filereader.filereader import FileReader
from ml_recon.dataset.filereader.nifti import NiftiFileReader
from ml_recon.dataset.sliceloader import SliceDataset
from ml_recon.utils import ifft_2d_img, fft_2d_img



class MultiContrastLoader(Dataset):
    """
    Takes data directory and creates a dataset. Before using you need to specify the file reader 
    to use in the filereader variable. 
    """
    filereader: FileReader

    def __init__(
            self,
            data_dir: Union[str, os.PathLike], 
            raw_sample_filter: Optional[Callable] = lambda _: True,  # if not defined let everything though
            transforms: Optional[Callable] = None,
            build_new_header: bool = False
            ):
        self.tranforms = transforms
        self.data_list = {}

        sample_dir = os.listdir(data_dir)
        sample_dir.sort()
        
        slices = []
        
        for sample in sample_dir:
            sample_path = os.path.join(data_dir, sample)
            modalities = os.listdir(sample_path)

            data_dict = {} 
            for i, modality in enumerate(modalities):
                # remove segmentation maps
                if 'seg' in modality:
                    continue

                modality_path = os.path.join(sample_path, modality)
                modality_name = modality.split('_')[-1].split('.')[0]
                data_dict[modality_name] = modality_path
                if i == 0:
                    with NiftiFileReader(modality_path) as fr:
                        slices = fr.shape[2]

        # cumulative sum is used to match index of slice to index of volume.
        self.volume_index_mapping = np.cumsum(slices)

    # length of dataset is the cumulative sum of the slices
    def __len__(self):
        return self.volume_index_mapping[-1]

    def __getitem__(self, index):
        volume_index, slice_index = self.get_vol_slice_index(index)
        images = self.get_data_from_indecies(volume_index, slice_index)
        images = self.resample(images)
        k_space = self.simulate_k_space(images)
        return k_space

    def get_vol_slice_index(self, index):
        volume_index = np.argmax(self.volume_index_mapping < index)
        slice_index = self.volume_index_mapping[index] - volume_index
        return volume_index,slice_index
    
    def get_data_from_indecies(self, volume_index, slice_index):
        files = self.data_list[volume_index]
        data = []
        for modality, file_name in files.items():
            with self.filereader(file_name) as fr:
                slice = fr[:, :, slice_index]
                data.append(slice)
        
        data = np.stack(data, axis=0)

        return data 

    def simulate_k_space(self, image):
        image_w_sense = self.apply_sensetivity(image)
        image_w_phase = self.generate_and_apply_phase(image_w_sense)
        k_space = ifft_2d_img(image_w_phase)
        k_space_w_noise = self.apply_noise(k_space)
        return k_space_w_noise

    def build_phase(self, nx, ny, center_region):
        phase_frequency = torch.zeros((nx, ny))
        center = (nx//2, ny//2)
        center_box_x = slice(center[0] - center_region//2, center[0] + np.ceil(center/2))
        center_box_y = slice(center[1] - center_region//2, center[1] + np.ceil(center/2))
        coeff = torch.rand(len(center_box_x), len(center_box_y))
        phase_frequency[center_box_x, center_box_y] = coeff

        phase = ifft_2d_img(phase_frequency)
        
        return phase

    def apply_phase_map(self, data, phase):
        data = data * torch.exp(1j* phase)
        return data

    def generate_and_apply_phase(self, data):
        center_region = 40
        phase = self.build_phase(data.shape[-1], data.shape[-2], center_region)
        data = self.apply_phase_map(data, phase)
        return data

    def apply_noise(self, k_space):
        rng = np.random.default_rng()
        noise = rng.normal(scale=1, size=k_space.shape) + 1j * rng.normal(scale=1, size=k_space.shape)
        k_space += noise
        return k_space


    
    def resample(self, data):
        resample_height = 128
        resample_width = 128 
        k_space = data['k_space']
        k_space = k_space[:, ::2, :]
        _, height, width = k_space.shape
        height_cent, width_cent = int(height/2), int(width/2)
        k_space = k_space[:, height_cent - resample_height//2:height_cent + resample_height//2, width_cent - resample_width//2: width_cent + resample_width//2]
        data['k_space'] = k_space

        recon = data['recon']
        xv, yv = np.meshgrid(np.linspace(0, recon.shape[0]-1, 128), np.linspace(0, recon.shape[1]-1, 128))
        recon_downsampled = interpn((range(recon.shape[0]), range(recon.shape[1])), recon, (yv, xv), bounds_error=False)
        data['recon'] = recon_downsampled
        
        return data

if __name__ == '__main__':
    MultiContrastLoader('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/')
