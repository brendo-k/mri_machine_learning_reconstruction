from ml_recon.dataset.filereader.read_h5 import H5FileReader
import os
import json
import xmltodict
import numpy as np

def move_16_chans(path, output=None):
    # check to make sure it is directory
    if path[-1] != '/':
        path += '/'

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    for file in files:
        print(file)
        full_path = os.path.join(path, file)
        try:
            with H5FileReader(full_path) as fr:
                # loop through all the slices
                for i_slice in range(fr['reconstruction_rss'].shape[0]):
                    coils = fr['kspace'].shape[1]
                    if coils == 16:
                        os.rename(full_path, os.path.join(path, '16_chans', file))
        except OSError:
            print(full_path)

    return 1
    
if __name__ == "__main__":
    move_16_chans("/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/")
