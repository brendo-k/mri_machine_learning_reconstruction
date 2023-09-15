import os
from typing import Callable, Optional, Union
from torch.utils.data import Dataset
import torch
from scipy.interpolate import interpn
import nibabel as nib
import numpy as np

from ml_recon.dataset.filereader.nifti import NiftiFileReader
from ml_recon.utils import fft_2d_img, ifft_2d_img
from ml_recon.dataset.undersample import gen_pdf_columns, calc_k

class MultiContrastLoader(Dataset):
    """
    Takes data directory and creates a dataset. Before using you need to specify the file reader 
    to use in the filereader variable. 
    """

    def __init__(
            self,
            data_dir: Union[str, os.PathLike], 
            nx:int = 256,
            ny:int = 256,
            R: int = 4, 
            R_hat: int = 2,
            poly_order: int = 8,
            acs_lines: int = 10,
            transforms: Optional[Callable] = None,
            ):
        self.tranforms = transforms

        sample_dir = os.listdir(data_dir)
        sample_dir.sort()
        self.slices = []
        self.data_list = []
        
        for sample in sample_dir:
            sample_path = os.path.join(data_dir, sample)
            modalities = os.listdir(sample_path)

            patient_dict = {} 
            for i, modality in enumerate(modalities):
                # remove segmentation maps
                if 'seg' in modality:
                    continue

                modality_path = os.path.join(sample_path, modality)
                modality_name = modality.split('_')[-1].split('.')[0]
                patient_dict[modality_name] = modality_path
                if i == 0:
                    with NiftiFileReader(modality_path) as fr:
                        self.slices.append(fr.shape[2])
            self.data_list.append(patient_dict)

        self.slices = np.array(self.slices)

        self.omega_prob = gen_pdf_columns(nx, ny, 1/R, poly_order, acs_lines)
        self.lambda_prob = gen_pdf_columns(nx, ny, 1/R_hat, poly_order, acs_lines)

        one_minus_eps = 1 - 1e-3
        self.lambda_prob[self.lambda_prob > one_minus_eps] = one_minus_eps

        self.k = torch.from_numpy(calc_k(self.lambda_prob, self.omega_prob)).float()
        

    # length of dataset is the sum of the slices
    def __len__(self):
        return sum(self.slices)

    def __getitem__(self, index):
        volume_index, slice_index = self.get_vol_slice_index(index)
        images = self.get_data_from_indecies(volume_index, slice_index)
        images = self.resample(images)
        k_space = self.simulate_k_space(images)
        k_space = torch.from_numpy(k_space)
        omega_mask = torch.zeros((k_space.shape[0], k_space.shape[-2], k_space.shape[-1]))
        return k_space

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
        files = self.data_list[volume_index]
        data = []
        modality_label = []
        for modality, file_name in files.items():
            file_object = nib.load(file_name) 
            image = file_object.get_fdata()
            slice = image[:, :, slice_index]
            data.append(slice)
            modality_label.append(modality)
        
        data = np.stack(data, axis=0)
        return data 

    def simulate_k_space(self, image):
        image_w_sense = self.apply_sensetivities(image)
        image_w_phase = self.generate_and_apply_phase(image_w_sense)
        k_space = fft_2d_img(image_w_phase)
        k_space = self.apply_noise(k_space)
        return k_space

    def apply_sensetivities(self, image):
        sense_map = np.load('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/sens.npy')
        sense_map = np.expand_dims(sense_map, 0).transpose((0, 3, 1, 2))
        image_sense = sense_map * np.expand_dims(image, 1)
        return image_sense      

    def build_phase(self, nx, ny, center_region):
        phase_frequency = np.zeros((nx, ny), dtype=np.complex64)
        center = (nx//2, ny//2)
        center_box_x = slice(center[0] - center_region//2, center[0] + np.ceil(center_region/2).astype(int))
        center_box_y = slice(center[1] - center_region//2, center[1] + np.ceil(center_region/2).astype(int))
        coeff = np.random.normal(size=(center_region, center_region)) + 1j * np.random.normal(size=(center_region, center_region))
        phase_frequency[center_box_x, center_box_y] = coeff

        phase = ifft_2d_img(phase_frequency)
        phase = np.angle(phase)
        
        return phase

    def apply_phase_map(self, data, phase):
        data *= np.exp(1j * phase)
        return data

    def generate_and_apply_phase(self, data):
        center_region = 10
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
        contrasts, height, width = data.shape
        y = np.arange(0, height)
        x = np.arange(0, width)
        c = np.arange(0, contrasts)

        yi = np.linspace(0, height - 1, resample_height)
        xi = np.linspace(0, width - 1, resample_width)
        (ci, yi, xi) = np.meshgrid(c, yi, xi, indexing='ij')

        new_data = interpn((c, y, x), data, (ci.flatten(), yi.flatten(), xi.flatten()))
        
        return np.reshape(new_data, (contrasts, resample_height, resample_width))


import matplotlib.pyplot as plt 
from ml_recon.utils import image_slices, root_sum_of_squares
if __name__ == '__main__':
    data = MultiContrastLoader('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/training_subset/')
    print(data[50].shape)
    image_slices(root_sum_of_squares(ifft_2d_img(data[100]), coil_dim=1), cmap='gray')
    plt.savefig('contrasts')
    image_slices(np.angle(ifft_2d_img(data[100])[0]), cmap='gray')
    plt.savefig('angle')
