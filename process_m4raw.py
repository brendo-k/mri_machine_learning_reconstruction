import h5py 
import numpy as np
import os
import shutil
from collections import defaultdict   

def main():
    path_to_train_dir = '/home/brenden/Downloads/M4RawV1.5_multicoil_train/multicoil_train'
    path_to_train_out = '/home/brenden/Downloads/train/'

    path_to_val_dir = '/home/brenden/Downloads/M4RawV1.5_multicoil_val/multicoil_val'
    path_to_val_out = '/home/brenden/Downloads/val/'

    path_to_test_dir = '/home/brenden/Downloads/M4Raw_multicoil_test/multicoil_test'
    path_to_test_out = '/home/brenden/Downloads/test/'


    #extract_first_average(path_to_val_dir, path_to_val_out)
    #extract_first_average(path_to_train_dir, path_to_train_out)
    extract_first_average(path_to_test_dir, path_to_test_out)


def extract_first_average(data_dir, out_dir):
    sample_scans = defaultdict(list)
    data_files = os.listdir(data_dir)
    for file in data_files:
        print(file)
        file_identitifes = file.split('_')[0]
        sample_scans[file_identitifes].append(file)
    
    for key in sample_scans.keys():
        contrast_files = []
        for file in sample_scans[key]:
            if file.endswith('01.h5'):
                print(file)
                contrast_files.append(file)
        contrast_files.sort()

        k_space_data = []
        ismrmd_header = []
        reconstruction_rss = []
        contrasts = []
        for file in contrast_files:
            with h5py.File(os.path.join(data_dir, file), 'r') as f:
                contrast = file.split('_')[1][:-5]
                contrasts.append(contrast.lower())
                k_space_data.append(f['kspace'][:])
                ismrmd_header.append(f['ismrmrd_header'][()])
                reconstruction_rss.append(f['reconstruction_rss'][:])
        
        k_data = np.stack(k_space_data, axis=0)
        chunk_size = k_data.shape
        chunk_size = (1, 1, chunk_size[2], chunk_size[3], chunk_size[4])
        img_data = np.stack(reconstruction_rss, axis=0)

        with h5py.File(os.path.join(out_dir, key + '.h5'), 'w') as f:
            f.create_dataset('kspace', data=k_data, chunks=chunk_size)
            f.create_dataset('reconstruction_rss', data=img_data)
            f.create_dataset('ismrmrd_header', data=ismrmd_header)
            f.create_dataset('contrasts', data=contrasts)
        

if __name__ == '__main__':
    main()