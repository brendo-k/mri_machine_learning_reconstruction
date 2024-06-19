from ml_recon.dataset.Brats_dataset import SimulatedBrats
import numpy as np
import nibabel as nib
import h5py
import os
import multiprocessing
from functools import partial
from itertools import repeat
from scipy.ndimage import rotate

IMAGE_SIZE = (256, 256)

# Define a function to process a single file
def process_file(file, out_path, seed):
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
    k_space = np.zeros((4, 4, IMAGE_SIZE[0], IMAGE_SIZE[1], (images.shape[-1] - 106)//3), dtype=np.complex64)
    for i in range(images.shape[-1]):
        if i < 70: 
            continue
        if i >= images.shape[-1]-36:
            break
        if i % 3 == 0:
            cur_images = SimulatedBrats.resample(images[..., i], IMAGE_SIZE[0], IMAGE_SIZE[1])
            cur_images = np.transpose(cur_images, (0, 2, 1))
            cur_images = rotate(cur_images, 45, axes=(-1, -2), reshape=False)
            k_space[..., (i-70)//3] = SimulatedBrats.simulate_k_space(
                                        cur_images, seed+i, same_phase=False, 
                                        center_region=10, noise_std=0.001, coil_size=4
                                        )

    k_space = np.transpose(k_space, (4, 0, 1, 2, 3)).astype(np.complex64)

    try:
        os.makedirs(os.path.join(out_path, patient_name))
    except FileExistsError as e:
        print(e)

    try:
        save_file = os.path.join(out_path, patient_name, patient_name + '.h5')
        with h5py.File(save_file, 'w') as fr:
            dset = fr.create_dataset("k_space", k_space.shape, dtype=np.complex64)
            dset[...] = k_space
            dset = fr.create_dataset("contrasts", data=modality_name)
        print(f'saved to file: {save_file}')

    except Exception as e:
        print(e)


    #np.save(file=os.path.join(out_path, patient_name, patient_name), arr=k_space)
    with open(os.path.join(out_path, patient_name, 'labels'), 'w') as f:
        f.write(','.join(modality_name))

    print(f'Done file {os.path.join(out_path, patient_name, patient_name + ".h5")}')


if __name__ == '__main__':
    dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/subset/'
    save_dir = '/home/kadotab/scratch/rotated/'
    dataset_splits = ['train', 'test', 'val']

    # Create a pool of worker processes
    num_processes = int(os.getenv('SLURM_CPUS_PER_TASK'))  # Adjust as needed
    print(num_processes)
    pool = multiprocessing.Pool(processes=num_processes)

    for split in dataset_splits:
        print(split)
        # Process each file in parallel
        results = []
        files = os.listdir(os.path.join(dir, split))
        files = [os.path.join(dir, split, file) for file in files]
        seeds = [np.random.randint(0, 1_000_000_000) for _ in range(len(files))]

        #for file in files:
        #    process_file(file, os.path.join(save_dir, split), np.random.randint(0, 1_000_000_000))

        pool.starmap(process_file, zip(files.__iter__(), repeat(os.path.join(save_dir, split)), seeds))
