import os
import csv
import time
from typing import Callable, Optional, Union, Collection
from argparse import ArgumentParser

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import rotate
import torch
import numpy as np

import nibabel as nib

from ml_recon.dataset.k_space_dataset import KSpaceDataset
from ml_recon.utils import fft_2d_img, ifft_2d_img, root_sum_of_squares
from ml_recon.utils.espirit import espirit

class SimulatedBrats(KSpaceDataset):
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
            extension: str = "nii.gz",
            deterministic: bool = False,
            center_region_phase: int = 20, 
            noise_std: float = 0.001
            ):
        assert contrasts, 'Contrast list should not be empty!'

        super().__init__(nx=nx, ny=ny)

        self.transforms = transforms
        self.contrasts = np.array([contrast.lower() for contrast in contrasts])
        self.extension = extension
        self.center_region_phase = center_region_phase
        self.noise_std = noise_std

        sample_dir = os.listdir(data_dir)
        sample_dir.sort()
        if deterministic:
            self.seed = 0
        else:
            self.seed = torch.seed()

        slices = []
        self.data_list = []
        
        start = time.time()
        for sample in sample_dir:
            sample_path = os.path.join(data_dir, sample)
            modalities = os.listdir(sample_path)

            patient_dict = {} 

            patient_dict = self.get_contrast_files(modalities, sample_path)
            slices.append(self.get_num_slices(patient_dict))

            self.data_list.append(patient_dict)

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
        images = self.resample(data, self.nx, self.ny)
        images = np.transpose(images, (0, 2, 1))
        random_index = index + self.seed
        data = SimulatedBrats.simulate_k_space(images, random_index,
                                               center_region=self.center_region_phase, 
                                               noise_std=self.noise_std)
        data = torch.from_numpy(data)

        if self.transforms:
            data = self.transforms(data)
        return data

    def get_contrast_files(self, modalities, sample_path):
        patient_dict = {}
        for modality in modalities:
            # skip segmentation maps
            if 'seg' in modality:
                continue
            if self.extension in modality:
                modality_path = os.path.join(sample_path, modality)
                modality_name = modality.split('_')[-1].split('.')[0]
                patient_dict[modality_name] = modality_path

        return patient_dict

    
    def get_num_slices(self, patient_dict):
        modality_path = patient_dict['t1']
        if self.extension == 'npy':
            slices = np.load(modality_path).shape[2]
        elif self.extension == 'nii.gz':
            slices = nib.nifti1.load(modality_path).get_fdata().shape[2]
        else:
            raise ValueError(f'no file reader for extension {self.extension}, only nii.gz and npy')
        assert slices > 0
        slices = (slices - 106)//3 # the first 70 slices and last 35 aren't useful and we take every third slice
        
        return slices
        

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
        
        slice_index *= 3 # since we are grabbing every third slice
        slice_index += 70 # since we remove the first 70 slices
        return volume_index, slice_index 

    
    def get_data_from_indecies(self, volume_index, slice_index):
        files = self.data_list[volume_index]
        data = []
        modality_label = []
        for modality in sorted(files):
            if modality.lower() in self.contrasts: 
                file_name = files[modality]
                ext = os.path.splitext(file_name)[1]
                if 'gz' in ext:
                    file_object = nib.nifti1.load(file_name) 
                    image = file_object.get_fdata()
                elif 'npy' in ext:
                    image = np.load(file_name)
                else:
                    raise ValueError(f'Can not load file with extention {ext}')

                slice = image[:, :, slice_index]
                slice = (slice - np.min(slice)) / (np.max(slice) - np.min(slice))
                slice = rotate(slice, 45, reshape=False)
                data.append(slice)
                modality_label.append(modality)
    
        data = np.stack(data, axis=0)
        return data, modality_label

    @staticmethod
    def simulate_k_space(image, seed, same_phase=False, center_region=20, noise_std=0.001, coil_size=8):
        #simulate some random motion
        rng = np.random.default_rng()
        x_shift, y_shift = rng.integers(-10, 10), rng.integers(-10, 10)
        image = np.roll(np.roll(image, x_shift, axis=-1), y_shift, axis=-2)
        #image [Contrast height width]
        #image_w_sense = SimulatedBrats.apply_sensetivities(image, coil_size)
        image_w_sense = image
        #image_w_sense [Contrast coil height width]
        image_w_phase = SimulatedBrats.generate_and_apply_phase(image_w_sense, seed, same_phase=same_phase, center_region=center_region)
        k_space = fft_2d_img(image_w_phase)
        k_space = SimulatedBrats.apply_noise(k_space, seed, noise_std)
        return k_space

    @staticmethod
    def apply_sensetivities(image, coil_size):
        coil_size = str(coil_size)
        sense_map = np.load('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/coil_compressed_' + coil_size + '.npy')
        #sense_map = np.squeeze(sense_map)
        sense_map = np.transpose(sense_map, (0, 2, 1))
        sense_map = sense_map[:, 25:-26, 25:-25]

        mag_sense_map = np.abs(sense_map)
        mag_sense_phase = np.angle(sense_map)

        resampled_sense_mag = SimulatedBrats.resample(mag_sense_map, image.shape[1], image.shape[2])
        resampled_sense_phase = SimulatedBrats.resample(mag_sense_phase, image.shape[1], image.shape[2])
        
        resampled_sense = resampled_sense_mag * np.exp(resampled_sense_phase * 1j)

        sense_map = np.expand_dims(resampled_sense, 0)
        image_sense = sense_map * np.expand_dims(image, 1)
        return image_sense      

    @staticmethod
    def generate_and_apply_phase(data, seed, center_region=20, same_phase=False):
        if same_phase: 
            nc = 1
        else:
            nc = data.shape[0]

        phase = SimulatedBrats.build_phase(center_region, data.shape[2], data.shape[3], nc, same_phase=same_phase, seed=seed)
        data = SimulatedBrats.apply_phase_map(data, phase)
        return data


    @staticmethod
    def build_phase(center_region, nx, ny, nc, same_phase=False, seed=None):
        rng = np.random.default_rng(seed)

        phase_frequency = np.zeros((1, nx, ny), dtype=np.complex64)
        if not same_phase:
            phase_frequency = np.tile(phase_frequency, (nc, 1, 1))

        center = (nx//2, ny//2)
        center_box_x = slice(center[0] - center_region//2, center[0] + np.ceil(center_region/2).astype(int))
        center_box_y = slice(center[1] - center_region//2, center[1] + np.ceil(center_region/2).astype(int))
        coeff = rng.random(size=(phase_frequency.shape[0], center_region, center_region)) + 1j * rng.random(size=(phase_frequency.shape[0], center_region, center_region))
        coeff -= 0.5 + 1j * 0.5
        phase_frequency[:, center_box_x, center_box_y] = coeff

        phase = fft_2d_img(phase_frequency)
        phase = np.angle(phase)
        
        return phase


    @staticmethod
    def apply_phase_map(data, phase):
        return data * np.exp(1j * np.expand_dims(phase, 1))


    @staticmethod
    def apply_noise(k_space, seed, noise_std):
        rng = np.random.default_rng(seed)
        noise_scale = noise_std
        noise = rng.normal(scale=noise_scale, size=k_space.shape) + 1j * rng.normal(scale=noise_scale, size=k_space.shape)
        k_space += noise
        return k_space


    @staticmethod
    def resample(data, resample_height, resample_width):
        contrasts, height, width = data.shape
        y = np.arange(0, height)
        x = np.arange(0, width)
        c = np.arange(0, contrasts)

        yi = np.linspace(0, height - 1, resample_height)
        xi = np.linspace(0, width - 1, resample_width)
        (ci, yi, xi) = np.meshgrid(c, yi, xi, indexing='ij')
        interpolator = RegularGridInterpolator((c, y, x), data, method='nearest')
        data = interpolator((ci.flatten(), yi.flatten(), xi.flatten()))
        interp_data = np.reshape(data, (contrasts, resample_height, resample_width))
        
        return interp_data


import matplotlib.pyplot as plt
from ml_recon.utils import root_sum_of_squares, ifft_2d_img, image_slices
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/subset/test/'
    dataset = SimulatedBrats(data_dir, nx=128, ny=128, center_region_phase=0)

    k_space = dataset[args.index]
    volume_index, slice_index = dataset.get_vol_slice_index(20)
    data, _ = dataset.get_data_from_indecies(volume_index, slice_index)

    fig, ax = plt.subplots(2,2)
    
    print(k_space.shape)
    for i in range(k_space.shape[0]):
        ax[i%2, i//2].imshow(root_sum_of_squares(ifft_2d_img(k_space[i]), coil_dim=0), cmap='gray')

    k_space = k_space.numpy()
    k_space = np.transpose(k_space, (0, 2, 3, 1)) # move coil dimension to last

    maps = espirit(k_space[[0], :, :, :], 8, 30, 0.01, 0.9)
    maps = maps[0, :, :, :, 0]

    imgs = ifft_2d_img(k_space, axes=(1, 2))

    combined_imgs = 1/(np.sum(np.conj(maps) * maps, axis=-1) + 1e-6) * np.sum(np.conj(maps) * imgs, axis=-1)

    fig, ax = plt.subplots(2,2)
    
    for i in range(combined_imgs.shape[0]):
        ax[i%2, i//2].imshow(np.angle(combined_imgs[i]), cmap='gray')

    plt.show()
