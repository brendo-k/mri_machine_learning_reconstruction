from ml_recon.utils.simulated_k_space_from_brats import simulate_k_space, resample
import numpy as np
import nibabel as nib
import torch
import h5py
import os
import multiprocessing
from itertools import repeat
from ml_recon.utils import fft_2d_img, ifft_2d_img, k_to_img
import sys
import gc

IMAGE_SIZE = (256, 256)

# Define a function to process a single file
def process_file(file, out_path, seed, noise, coil_file):
    print(f'Starting file {file}, with seed: {seed}')
    patient_name = file.split('/')[-1]

    modality_files = os.listdir(file)
    modality_files.sort()
    modality_name = []
    
    images = []
    for modality in modality_files:
        if 'nii' in modality and 'seg' not in modality:
            modality_name.append(modality.split('_')[-1].split('.')[0])
            contrast_image = nib.nifti1.load(os.path.join(dir, file, modality)).get_fdata()
            images.append(contrast_image)
        
    images = np.stack(images, axis=0)
    images = images/np.max(images, axis=(1, 2), keepdims=True)
    k_space = np.zeros((4, int(10), IMAGE_SIZE[0], IMAGE_SIZE[1], (images.shape[-1] - 106)//3), dtype=np.complex64)
    gt_img = np.zeros((4, IMAGE_SIZE[0], IMAGE_SIZE[1], (images.shape[-1] - 106)//3))
    for i in range(images.shape[-1]):
        if i < 70: 
            continue
        if i >= images.shape[-1]-36:
            break
        if i % 3 == 0:
            cur_images = images[..., i]

            cur_images = np.transpose(cur_images, (0, 2, 1))
            cur_images = resample(cur_images, 256, 256, 'linear')
            sim_k_space, gt = simulate_k_space(
                                        cur_images, 
                                        seed+i,
                                        noise_std=noise, 
                                        coil_file=coil_file
                                        )
            k_space[..., (i-70)//3] = sim_k_space
            gt_img[..., (i-70)//3] = gt

    k_space = np.ascontiguousarray(np.transpose(k_space, (4, 0, 1, 2, 3)).astype(np.complex64))
    gt_img = np.ascontiguousarray(np.transpose(gt_img, (3, 0, 1, 2)))

    try:
        os.makedirs(os.path.join(out_path, patient_name))
    except FileExistsError as e:
        print(e)

    try:
        save_file = os.path.join(out_path, patient_name, patient_name + '.h5')
        chunk_size = (1, 1, 10, IMAGE_SIZE[0], IMAGE_SIZE[1])
        with h5py.File(save_file, 'w') as fr:
            dset = fr.create_dataset("k_space", k_space.shape, dtype=np.complex64, chunks=chunk_size)
            dset[...] = k_space
            dset = fr.create_dataset("contrasts", data=modality_name)
            dset = fr.create_dataset("reconstruction_rss", data=k_to_img(torch.from_numpy(k_space), coil_dim=2))
            dset = fr.create_dataset("ground_truth", data=gt_img)
        print(f'saved to file: {save_file}')
        del fr, dset, gt_img, modality_name
        gc.collect()

    except Exception as e:
        print(e)


    print(f'Done file {os.path.join(out_path, patient_name, patient_name + ".h5")}')


if __name__ == '__main__':
    dir = '/home/brenden/Documents/Coding/data/subset'
    save_dir = '/home/kadotab/scratch/sim_subset'
    dataset_splits = ['train', 'test', 'val']

    noise = float(sys.argv[1])
    save_dir = str(sys.argv[2])
    coil_file = str(sys.argv[3])

    # Create a pool of worker processes
    num_processes = 4
    print(num_processes)
    pool = multiprocessing.Pool(processes=num_processes)

    for split in dataset_splits:
        print(split)
        # Process each file in parallel
        files = os.listdir(os.path.join(dir, split))
        files = [os.path.join(dir, split, file) for file in files]
        #files = [files[i] for i in range(20)]
        seeds = [np.random.randint(0, 1_000_000_000) for _ in range(len(files))]
        for file, seed in zip(files, seeds):
             process_file(
                 file, 
                 os.path.join(save_dir, split),
                 seed, 
                 noise, 
                 coil_file
             )

        #pool.starmap(process_file, 
        #            zip(
        #                files.__iter__(), 
        #                repeat(os.path.join(save_dir, split)),
        #                seeds,
        #                repeat(noise),
        #                repeat(coil_file)
        #                )
        #            )
