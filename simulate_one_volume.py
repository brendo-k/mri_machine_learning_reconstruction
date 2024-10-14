from ml_recon.dataset.Brats_dataset import SimulatedBrats
import numpy as np
import nibabel as nib
import h5py
import os
import multiprocessing
from itertools import repeat
from ml_recon.utils import fft_2d_img, ifft_2d_img, root_sum_of_squares
import matplotlib.pyplot as plt
from torchvision.transforms.functional import center_crop
import torch
from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.dataset.undersample import get_mask_from_distribution, gen_pdf


IMAGE_SIZE = (128, 128)
COIL_SIZE = 12

# Define a function to process a single file
def process_file(file, seed):
    print(f'Starting file {file}, with seed: {seed}')
    patient_name = file.split('/')[-1]

    modality_files = os.listdir(file)
    modality_files.sort()
    modality_name = []
    
    images = []
    for modality in modality_files:
        if 'nii' in modality and 'seg' not in modality:
            modality_name.append(modality.split('_')[-1].split('.')[0])
            images.append(nib.nifti1.load(os.path.join(dir, file, modality)).get_fdata())
        
    images = np.stack(images, axis=0)
    k_space = np.zeros((4, COIL_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], (images.shape[-1] - 106)//3), dtype=np.complex64)
    i = 100
    #cur_images = SimulatedBrats.resample(images[..., i], IMAGE_SIZE[0], IMAGE_SIZE[1])
    cur_images = images[..., i]
    cur_images = fft_2d_img(cur_images)
    _, y, x = cur_images.shape 
    y_start = y//2 - IMAGE_SIZE[0]//2
    x_start = x//2 - IMAGE_SIZE[1]//2
    cur_images = cur_images[:, y_start:y_start + IMAGE_SIZE[0], x_start:x_start + IMAGE_SIZE[1]]
    cur_images = ifft_2d_img(cur_images)

    cur_images = np.transpose(cur_images, (0, 2, 1))
    k_space = SimulatedBrats.simulate_k_space(
                                cur_images, seed+i, same_phase=False, 
                                center_region=0, noise_std=0.0001, coil_size=COIL_SIZE
                                )
    pdf = gen_pdf(False, 128, 128, 1/4, 8, 10)
    mask = get_mask_from_distribution(pdf, np.random.default_rng(), False)

    undersampled_kspace = k_space * mask[None, None, :, :]
    images = root_sum_of_squares(ifft_2d_img(k_space), coil_dim=1)
    images_undersampled = root_sum_of_squares(ifft_2d_img(undersampled_kspace), coil_dim=1)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(np.abs(images[0, :, :]))
    ax[0, 1].imshow(np.abs(images_undersampled[0, :, :]))
    ax[1, 0].imshow(np.real(images_undersampled[0, :, :]))
    ax[1, 1].imshow(np.imag(images_undersampled[0, :, :]))
    plt.show()
    

if __name__ == '__main__':
    dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/subset/train/'
    files = os.listdir(dir)
    process_file(os.path.join(dir, files[0]), 42)
