from typing import Callable
import os
import json
import numpy as np

from ml_recon.dataset.filereader.filereader import FileReader

def make_header(path, filereader: FileReader, output=None, sample_filter:Callable = lambda _ : True):
    # check to make sure it is directory
    path = os.path.join(path, '')

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files.sort()

    header_index = {}
    index = 0
    for file_number, file in enumerate(files):
        full_path = os.path.join(path, file)
        try:
            with filereader(full_path) as fr:
                # loop through all the slices
                if sample_filter(fr):
                    for i_slice in range(fr['recon'].shape[0]):
                        header_index[index] = {
                            'slice_index': i_slice,
                            'file_name': full_path,
                            'file_number': file_number
                        }
                        if 'coils' in fr.keys():
                            header_index[index]['coils'] = fr['coils']
                        else:
                            header_index[index]['coils'] = 1

                        index += 1
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
