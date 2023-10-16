import os
import csv
import time
from typing import Callable, Optional, Union, Collection
from argparse import ArgumentParser

from scipy.interpolate import interpn
import torch
import numpy as np

import nibabel as nib

from ml_recon.dataset.k_space_dataset import KSpaceDataset
from ml_recon.utils import fft_2d_img, ifft_2d_img, root_sum_of_squares

class BratsDataset(KSpaceDataset):
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
            extension: str = "npy"
            ):
        assert contrasts, 'Contrast list should not be empty!'

        super().__init__(nx=nx, ny=ny)

        self.transforms = transforms
        self.contrasts = np.array([contrast.lower() for contrast in contrasts])
        self.extension = extension

        sample_dir = os.listdir(data_dir)
        sample_dir.sort()

        slices = []
        self.data_list = []
        
        start = time.time()
        for sample in sample_dir:
            sample_path = os.path.join(data_dir, sample)
            modalities = os.listdir(sample_path)
            numpy_files = [file for file in modalities if file.endswith('.npy')]
            num_npy = len(numpy_files)

            patient_dict = {} 

            if num_npy == 1:
                self.simulated = True
                patient_dict['all'] = os.path.join(sample_path, numpy_files[0])
                num_slices = np.load(patient_dict['all']).shape[-1]
            else:
                self.simulated = False
                self.seed = np.random.randint(0, 100_000)
                patient_dict, num_slices = self.get_contrast_files(modalities, sample_path)

            slices.append(num_slices)
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
        if not self.simulated:
            images = self.resample(data)
            images = np.transpose(images, (0, 2, 1))
            data = self.simulate_k_space(images)
        data = torch.from_numpy(data)

        if self.transforms:
            data = self.transforms(data)
        return data

    def get_contrast_files(self, modalities, sample_path):
        patient_dict = {}
        first_file = True
        slices = -1
        for modality in modalities:
            # skip segmentation maps
            if 'seg' in modality:
                continue
            if self.extension in modality:
                modality_path = os.path.join(sample_path, modality)
                modality_name = modality.split('_')[-1].split('.')[0]
                patient_dict[modality_name] = modality_path

                # get number of slices
                if first_file: 
                    first_file = False

                    if self.extension == 'npy':
                        slices = np.load(modality_path).shape[2]
                    elif self.extension == 'nii.gz':
                        slices = nib.nifti1.load(modality_path).get_fdata().shape[2]
                    else:
                        raise ValueError(f'no file reader for extension {self.extension}, only nii.gz and npy')
        assert slices > 0

        slices = slices - 90 # the first 70 slices and last 20 aren't useful
        return patient_dict, slices

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
        
        if not self.simulated: 
            slice_index += 70
        return volume_index, slice_index 
    
    def get_data_from_indecies(self, volume_index, slice_index):
        files = self.data_list[volume_index]
        data = []
        modality_label = []
        if 'all' in files.keys():
            data = np.load(files['all'])
            data = data[..., slice_index]

            labels_path = os.path.join(os.path.split(files['all'])[0], 'labels')
            with open(labels_path, 'r') as fr:
                reader = csv.reader(fr)
                all_labels = np.array(list(reader))
                all_labels = np.squeeze(all_labels)

            use_modality_index = np.isin(all_labels, self.contrasts)
            data = data[use_modality_index, ...]
            modality_label = all_labels[use_modality_index]
            
        else:
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
                    data.append(slice)
                    modality_label.append(modality)
        
            data = np.stack(data, axis=0)
        return data, modality_label

    def simulate_k_space(self, image):
        image_w_sense = self.apply_sensetivities(image)
        image_w_phase = self.generate_and_apply_phase(image_w_sense)
        k_space = fft_2d_img(image_w_phase)
        k_space = self.apply_noise(k_space)
        return k_space

    def apply_sensetivities(self, image):
        sense_map = np.load('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/sens.npy')
        sense_map = sense_map.transpose((2, 0, 1))
        sense_map = self.resample(sense_map)

        sense_map = np.expand_dims(sense_map, 0)
        image_sense = sense_map * np.expand_dims(image, 1)
        return image_sense      

    def generate_and_apply_phase(self, data):
        center_region = 6
        phase = self.build_phase(center_region)
        data = self.apply_phase_map(data, phase)

        return data
    def build_phase(self, center_region):
        nx = self.nx
        ny = self.ny
        phase_frequency = np.zeros((nx, ny), dtype=np.complex64)
        center = (nx//2, ny//2)
        center_box_x = slice(center[0] - center_region//2, center[0] + np.ceil(center_region/2).astype(int))
        center_box_y = slice(center[1] - center_region//2, center[1] + np.ceil(center_region/2).astype(int))
        rng = np.random.default_rng(self.seed)
        coeff = rng.normal(size=(center_region, center_region)) + 1j * rng.normal(size=(center_region, center_region))
        phase_frequency[center_box_x, center_box_y] = coeff

        phase = fft_2d_img(phase_frequency)
        phase = np.angle(phase)
        
        return phase

    def apply_phase_map(self, data, phase):
        data *= np.exp(1j * phase)
        return data


    def apply_noise(self, k_space):
        rng = np.random.default_rng(self.seed)
        noise_scale = 10
        noise = rng.normal(scale=noise_scale, size=k_space.shape) + 1j * rng.normal(scale=noise_scale, size=k_space.shape)
        k_space += noise
        return k_space


    def resample(self, data):
        resample_height = self.ny
        resample_width = self.nx 
        contrasts, height, width = data.shape
        y = np.arange(0, height)
        x = np.arange(0, width)
        c = np.arange(0, contrasts)

        yi = np.linspace(0, height - 1, resample_height)
        xi = np.linspace(0, width - 1, resample_width)
        (ci, yi, xi) = np.meshgrid(c, yi, xi, indexing='ij')

        new_data = interpn((c, y, x), data, (ci.flatten(), yi.flatten(), xi.flatten()))
        
        assert isinstance(new_data, np.ndarray)
        return np.reshape(new_data, (contrasts, resample_height, resample_width))

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = KSpaceDataset.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False)

        parser.add_argument(
                '--data_dir', 
                type=str, 
                default='/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/subset/', 
                help=''
                )

        parser.add_argument(
                '--contrasts', 
                type=str, 
                nargs='+',
                default=['t1', 't2', 'flair', 't1ce'], 
                help=''
                )

        return parser

from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser = BratsDataset.add_model_specific_args(parser)
    args = parser.parse_args()
    dataset = BratsDataset(os.path.join(args.data_dir, 'train'), contrasts=args.contrasts, extension='nii.gz')
    dataset = UndersampleDecorator(dataset)

    i = dataset[0]
    image = ifft_2d_img(i[2])
    image = root_sum_of_squares(image[0], coil_dim=0)
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.savefig('image')
