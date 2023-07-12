from ml_recon.dataset.filereader.read_h5 import H5FileReader
import os
import json
import xmltodict
import numpy as np

def make_header(path, output=None):
    # check to make sure it is directory
    if path[-1] != '/':
        path += '/'

    if output == None:
        output = path + 'header.json'

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    header_index = {}
    index = 0
    for file_number, file in enumerate(files):
        full_path = os.path.join(path, file)
        try:
            with H5FileReader(full_path) as fr:
                # loop through all the slices
                for i_slice in range(fr['reconstruction_rss'].shape[0]):
                    header = xmltodict.parse(np.array(fr['ismrmrd_header'], dtype=bytes))
                    header_index[index] = {
                        'slice_index': i_slice,
                        'file_name': full_path,
                        'coils': fr['kspace'].shape[1],
                        'T': header['ismrmrdHeader']['acquisitionSystemInformation']['systemFieldStrength_T'],
                        'file_number': file_number
                    }
                    index += 1
        except OSError:
            print(full_path)
    
    with open(output, 'w') as fp:
        json.dump(header_index, fp, indent=4)
    
    return(output)


if __name__ == "__main__":
    make_header('/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/multicoil_val/', '/home/kadotab/val.json')
    make_header('/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/multicoil_test/', '/home/kadotab/test.json')
    make_header('/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/multicoil_train/', '/home/kadotab/train.json')
