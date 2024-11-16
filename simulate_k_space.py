from ml_recon.utils.simulated_k_space_from_brats import simulate_k_space
import numpy as np
import nibabel as nib
import h5py
import os
import multiprocessing
from itertools import repeat
from ml_recon.utils import fft_2d_img, ifft_2d_img
import sys

IMAGE_SIZE = (240, 240)

# Define a function to process a single file
def process_file(file, out_path, seed, noise, coils):
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
    k_space = np.zeros((4, int(coils), IMAGE_SIZE[0], IMAGE_SIZE[1], (images.shape[-1] - 106)//3), dtype=np.complex64)
    for i in range(images.shape[-1]):
        if i < 70: 
            continue
        if i >= images.shape[-1]-36:
            break
        if i % 3 == 0:
            #cur_images = SimulatedBrats.resample(images[..., i], IMAGE_SIZE[0], IMAGE_SIZE[1])
            cur_images = images[..., i]
            cur_images = fft_2d_img(cur_images)
            _, y, x = cur_images.shape 
            y_start = y//2 - IMAGE_SIZE[0]//2
            x_start = x//2 - IMAGE_SIZE[1]//2
            cur_images = cur_images[:, y_start:y_start + IMAGE_SIZE[0], x_start:x_start + IMAGE_SIZE[1]]
            cur_images = ifft_2d_img(cur_images)

            cur_images = np.transpose(cur_images, (0, 2, 1))
            k_space[..., (i-70)//3] = simulate_k_space(
                                        cur_images, seed+i,
                                        center_region=20, noise_std=noise, coil_size=coils
                                        )

    k_space = np.ascontiguousarray(np.transpose(k_space, (4, 0, 1, 2, 3)).astype(np.complex64))

    try:
        os.makedirs(os.path.join(out_path, patient_name))
    except FileExistsError as e:
        print(e)

    try:
        save_file = os.path.join(out_path, patient_name, patient_name + '.h5')
        chunk_size = (1, 1, int(coils), IMAGE_SIZE[0], IMAGE_SIZE[1])
        with h5py.File(save_file, 'w') as fr:
            dset = fr.create_dataset("k_space", k_space.shape, dtype=np.complex64, chunks=chunk_size)
            dset[...] = k_space
            dset = fr.create_dataset("contrasts", data=modality_name)
        print(f'saved to file: {save_file}')

    except Exception as e:
        print(e)


    print(f'Done file {os.path.join(out_path, patient_name, patient_name + ".h5")}')
    return k_space


if __name__ == '__main__':
    dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/subset/'
    save_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset_random_phase/'
    dataset_splits = ['train', 'test', 'val']

    save_dir = sys.argv[1]
    noise = float(sys.argv[2])
    coils = sys.argv[3]

    # Create a pool of worker processes
    num_processes = int(os.getenv('SLURM_CPUS_PER_TASK'))  # Adjust as needed
    #num_processes = 1
    print(num_processes)
    pool = multiprocessing.Pool(processes=num_processes)

    for split in dataset_splits:
        print(split)
        # Process each file in parallel
        files = os.listdir(os.path.join(dir, split))
        files = [os.path.join(dir, split, file) for file in files]
        seeds = [np.random.randint(0, 1_000_000_000) for _ in range(len(files))]

        #for file in files:
        #    process_file(file, os.path.join(save_dir, split), np.random.randint(0, 1_000_000_000))

        #k_space = process_file(files[0], os.path.join(save_dir, split), seeds[0])
        #fig, ax = plt.subplots(2, 2)
        #ax[0, 0].imshow(root_sum_of_squares(ifft_2d_img(center_crop(torch.from_numpy(k_space[0, 0, :, :, :]), 128)), coil_dim = 0))
        #ax[1, 0].imshow(root_sum_of_squares(ifft_2d_img(center_crop(torch.from_numpy(k_space[0, 1, :, :, :]), 128)), coil_dim = 0))
        #ax[0, 1].imshow(root_sum_of_squares(ifft_2d_img(center_crop(torch.from_numpy(k_space[0, 2, :, :, :]), 128)), coil_dim = 0))
        #ax[1, 1].imshow(root_sum_of_squares(ifft_2d_img(center_crop(torch.from_numpy(k_space[0, 3, :, :, :]), 128)), coil_dim = 0))
        #plt.show()
        pool.starmap(process_file, 
                     zip(
                         files.__iter__(), 
                         repeat(os.path.join(save_dir, split)),
                         seeds,
                         repeat(noise),
                         repeat(coils))
                     )
