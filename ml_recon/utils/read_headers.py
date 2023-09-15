from typing import Callable
import os
import json
import h5py


def make_header(path, output=None, sample_filter:Callable = lambda _ : True):
    # check to make sure it is directory
    path = os.path.join(path, '')

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files.sort()

    header_index = []
    for file in files:
        full_path = os.path.join(path, file)
        if 'json' in file: 
            continue
        try:
            with h5py.File(full_path) as fr:
                # loop through all the slices
                if sample_filter(fr):
                    slices = fr['kspace'].shape[0]
                    header_index.append({
                        'slices': slices,
                        'file_name': full_path,
                    })
                    if 'coils' in fr.keys():
                        header_index[-1]['coils'] = fr['coils']
                    else:
                        header_index[-1]['coils'] = 1

        except OSError:
            print(full_path)
    if output:    
        with open(output, 'w') as fp:
            json.dump(header_index, fp, indent=4)
    
    return header_index


if __name__ == "__main__":
    make_header('/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/multicoil_val/', '/home/kadotab/val.json')
    #make_header('/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/multicoil_test/', '/home/kadotab/test.json')
    make_header('/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/multicoil_train/', '/home/kadotab/train.json')
