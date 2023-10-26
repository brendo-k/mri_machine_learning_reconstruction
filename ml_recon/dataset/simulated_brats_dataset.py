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

            patient_dict = {} 

            self.seed = np.random.randint(0, 100_000)
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
        data = SimulatedBrats.simulate_k_space(images, index + self.seed)
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
        slices = slices - 105 # the first 70 slices and last 35 aren't useful
        
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

    @staticmethod
    def simulate_k_space(image, seed):
        image_w_sense = SimulatedBrats.apply_sensetivities(image)
        image_w_phase = SimulatedBrats.generate_and_apply_phase(image_w_sense, seed)
        k_space = fft_2d_img(image_w_phase)
        k_space = SimulatedBrats.apply_noise(k_space, seed)
        return k_space

    @staticmethod
    def apply_sensetivities(image):
        sense_map = np.load('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/sens.npy')
        sense_map = sense_map.transpose((2, 0, 1))
        sense_map = SimulatedBrats.resample(sense_map, image.shape[1], image.shape[2])

        sense_map = np.expand_dims(sense_map, 0)
        image_sense = sense_map * np.expand_dims(image, 1)
        return image_sense      

    @staticmethod
    def generate_and_apply_phase(data, seed, center_region=6):
        phase = SimulatedBrats.build_phase(center_region, data.shape[2], data.shape[3], seed)
        data = SimulatedBrats.apply_phase_map(data, phase)
        return data


    @staticmethod
    def build_phase(center_region, nx, ny, seed):
        phase_frequency = np.zeros((nx, ny), dtype=np.complex64)
        center = (nx//2, ny//2)
        center_box_x = slice(center[0] - center_region//2, center[0] + np.ceil(center_region/2).astype(int))
        center_box_y = slice(center[1] - center_region//2, center[1] + np.ceil(center_region/2).astype(int))
        rng = np.random.default_rng(seed)
        coeff = rng.normal(size=(center_region, center_region)) + 1j * rng.normal(size=(center_region, center_region))
        phase_frequency[center_box_x, center_box_y] = coeff

        phase = fft_2d_img(phase_frequency)
        phase = np.angle(phase)
        phase = phase.astype(float)
        
        return phase


    @staticmethod
    def apply_phase_map(data, phase):
        return data * np.exp(1j * phase)


    @staticmethod
    def apply_noise(k_space, seed):
        rng = np.random.default_rng(seed)
        mean = np.mean(np.abs(k_space))
        noise_scale = mean * 0.05
        noise = rng.normal(scale=noise_scale, size=k_space.shape) + 1j * rng.normal(scale=noise_scale, size=k_space.shape)
        k_space += noise
        return k_space


    @staticmethod
    def resample(data, nx, ny):
        resample_height = ny
        resample_width = nx 
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


from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser = BratsDataset.add_model_specific_args(parser)
    args = parser.parse_args()
    dataset = SimulatedBrats(os.path.join(args.data_dir, 'train'), contrasts=args.contrasts, extension='nii.gz')
    dataset = UndersampleDecorator(dataset)

    i = dataset[0]
    image = ifft_2d_img(i[2])
    image = root_sum_of_squares(image[0], coil_dim=0)
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.savefig('image')
