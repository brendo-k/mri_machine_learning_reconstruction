from ml_recon.utils.simulated_k_space_from_brats import simulate_k_space, resample
from ml_recon.utils import k_to_img
import numpy as np
from nibabel import nifti1
import torch
import h5py
import os
import multiprocessing
from itertools import repeat
import sys
import gc
from pathlib import Path

IMAGE_SIZE = (240, 240)
DATASET_VARIANTS = {
    "baseline": {"noise": None, "phase_mode": "variable"},
    "no_noise": {"noise": 0.0, "phase_mode": "variable"},
    "no_noise_same_phase": {"noise": 0.0, "phase_mode": "shared"},
    "no_noise_no_phase": {"noise": 0.0, "phase_mode": "none"},
}

# Define a function to process a single file
def process_file(file, out_path, seed, noise, phase_mode, coil_file, num_coils):
    patient_name = Path(file).name

    modality_files = os.listdir(file)
    modality_files.sort()
    modality_name = []
    
    images = []
    for modality in modality_files:
        if 'nii' in modality and 'seg' not in modality:
            modality_name.append(modality.split('_')[-1].split('.')[0])
            contrast_image = nifti1.load(os.path.join(file, modality)).get_fdata()
            images.append(contrast_image)
        
    images = np.stack(images, axis=0)
    # Replace zeros with NaN
    images_nozero = np.where(images == 0, np.nan, images)
    # Compute median over non-zero values
    med = np.nanmedian(images_nozero, axis=(1, 2), keepdims=True)
    # Normalize
    images = images / med
    print(IMAGE_SIZE[0])
    k_space = np.zeros((4, num_coils, IMAGE_SIZE[0], IMAGE_SIZE[1], (images.shape[-1] - 100)//5), dtype=np.complex64)
    gt_img = np.zeros((4, IMAGE_SIZE[0], IMAGE_SIZE[1], (images.shape[-1] - 100)//5))
    for i in range(images.shape[-1]):
        if i < 70: 
            continue
        if i >= images.shape[-1]-30:
            break
        if i % 5 == 0:
            cur_images = images[..., i]
            cur_images = resample(cur_images, IMAGE_SIZE[0], IMAGE_SIZE[1], 'linear')
            cur_images = np.transpose(cur_images, (0, 2, 1))
            sim_k_space, gt = simulate_k_space(
                                        cur_images, 
                                        seed+i,
                                        noise_std=noise, 
                                        coil_file=coil_file,
                                        phase_mode=phase_mode,
                                        )
            k_space[..., (i-70)//5] = sim_k_space
            gt_img[..., (i-70)//5] = gt

    k_space = np.ascontiguousarray(np.transpose(k_space, (4, 0, 1, 2, 3)).astype(np.complex64))
    gt_img = np.ascontiguousarray(np.transpose(gt_img, (3, 0, 1, 2)))

    try:
        os.makedirs(os.path.join(out_path, patient_name))
    except FileExistsError as e:
        print(e)

    try:
        save_file = os.path.join(out_path, patient_name, patient_name + '.h5')
        chunk_size = (1, 1, num_coils, IMAGE_SIZE[0], IMAGE_SIZE[1])
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
    dataset_splits = ['train', 'test', 'val']

    dir = str(sys.argv[1])
    save_dir = str(sys.argv[2])
    coil_file = str(sys.argv[3])
    baseline_noise = float(sys.argv[4]) if len(sys.argv) > 4 else 0.001

    
    maps = np.load(coil_file)
    num_coils = maps.shape[0]

    # Create a pool of worker processes
    num_processes = 4
    print(f'Running with {num_processes} processes')
    print(f'Base save dir is: {save_dir}')
    pool = multiprocessing.Pool(processes=num_processes)

    for variant_name, variant_config in DATASET_VARIANTS.items():
        noise = baseline_noise if variant_config["noise"] is None else variant_config["noise"]
        phase_mode = variant_config["phase_mode"]
        variant_root = os.path.join(save_dir, variant_name)

        print(f'Generating variant: {variant_name}')

        for split in dataset_splits:
            print(split)

            # Process each file in parallel
            files = os.listdir(os.path.join(dir, split))
            files = [os.path.join(dir, split, file) for file in files]
            seeds = [np.random.randint(0, 1_000_000_000) for _ in range(len(files))]

            pool.starmap(process_file, 
                        zip(
                            files.__iter__(), 
                            repeat(os.path.join(variant_root, split)),
                            seeds,
                            repeat(noise),
                            repeat(phase_mode),
                            repeat(coil_file),
                            repeat(num_coils)
                            )
                        )
    pool.close()
    pool.join()
