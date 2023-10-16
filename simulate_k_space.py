from ml_recon.dataset.Brats_dataset import BratsDataset
import numpy as np
import nibabel as nib
import h5py
import os
import multiprocessing
from functools import partial
from itertools import repeat

# Define a function to process a single file
def process_file(file, out_path):
    print(f'Starting file {file}')
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
    k_space = np.zeros((4, 8, 256, 256, images.shape[-1] - 90), dtype=np.complex64)
    for i in range(images.shape[-1]):
        if i < 70: 
            continue
        if i >= images.shape[-1]-20:
            break
        cur_images = dataset_test.resample(images[..., i])
        k_space[..., i-70] = dataset_test.simulate_k_space(cur_images)

    k_space = np.transpose(k_space, (4, 0, 1, 2, 3))
    try:
        os.mkdir(os.path.join(out_path, patient_name))
    except FileExistsError as e:
        print(e)

    with h5py.File(os.path.join(out_path, patient_name, patient_name + '.h5'), 'w') as fr:
        dset = fr.create_dataset("k_space", k_space.shape, dtype=np.csingle)
        dset[...] = k_space
        dset = fr.create_dataset("contrasts", data=modality_name)

    #np.save(file=os.path.join(out_path, patient_name, patient_name), arr=k_space)

    with open(os.path.join(out_path, patient_name, 'labels'), 'w') as f:
        f.write(','.join(modality_name))

    print(f'Done file {file}')


if __name__ == '__main__':
    dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/subset/'
    save_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/unchunked/'
    dataset_splits = ['train', 'test', 'val']

    dataset_test = BratsDataset(os.path.join(dir, 'test'), nx=256, ny=256)

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
        print(files)
        
        pool.starmap(process_file, zip(files.__iter__(), repeat(os.path.join(save_dir, split))))
