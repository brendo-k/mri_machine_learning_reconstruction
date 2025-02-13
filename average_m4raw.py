import h5py 
import numpy as np
import os
import shutil
from collections import defaultdict   

def main():
    path_to_train_dir = '/home/brenden/Downloads/M4RawV1.5_multicoil_train/multicoil_train'
    path_to_train_out = '/home/brenden/Downloads/M4Raw_Averaged/train/'

    path_to_val_dir = '/home/brenden/Downloads/M4RawV1.5_multicoil_val/multicoil_val'
    path_to_val_out = '/home/brenden/Downloads/M4Raw_Averaged/val/'

    path_to_test_dir = '/home/brenden/Downloads/M4Raw_multicoil_test/multicoil_test'
    path_to_test_out = '/home/brenden/Downloads/M4Raw_Averaged/test/'


    average_images(path_to_val_dir, path_to_val_out)
    average_images(path_to_train_dir, path_to_train_out)
    average_images(path_to_test_dir, path_to_test_out)


def average_images(data_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sample_scans = defaultdict(list)
    data_files = os.listdir(data_dir)
    for file in data_files:
        print(file)
        file_identitifes = file.split('_')[0]
        sample_scans[file_identitifes].append(file)
    
    for key in sample_scans.keys():
        contrast_images = defaultdict(list)
        for file in sample_scans[key]:
            contrast = file.split('_')[1][:-5]
            with h5py.File(os.path.join(data_dir, file), 'r') as f:
                contrast = file.split('_')[1][:-5].lower()
                contrast_images[contrast].append(f['reconstruction_rss'][:])
        
        # this is probably bad! Don't use this  

        flair = np.stack(contrast_images['flair'], axis=0).mean(0)
        t1 = np.stack(contrast_images['t1'], axis=0).mean(0)
        t2 = np.stack(contrast_images['t2'], axis=0).mean(0)

        images = np.stack([flair, t1, t2], axis=0)

        with h5py.File(os.path.join(out_dir, key + '.h5'), 'w') as f:
            f.create_dataset('reconstruction_rss', data=images)
            f.create_dataset('contrasts', data=['flair', 't1', 't2'])
        

if __name__ == '__main__':
    main()