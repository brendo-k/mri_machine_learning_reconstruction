import h5py
import numpy as np

class H5FileReader():
    def __init__(self, file_name):
        self.file_name = file_name

    def __enter__(self):
        self.file_object = h5py.File(self.file_name, 'r')
        #header = xmltodict.parse(np.array(self.file_object['ismrmrd_header'], dtype=bytes))
        #data_object = {
        #        'coils': self.file_object['kspace'].shape[1],
        #        'kspace': self.file_object['kspace'],
        #        'T': header['ismrmrdHeader']['acquisitionSystemInformation']['systemFieldStrength_T'],
        #        'recon': self.file_object['reconstruction_rss']
        #        }

        return np.array(self.file_object['kspace'])

    def get_keys(self):
        return self.file_object.keys()  

    def __exit__(self, exec_type, exec_value, traceback):
        if exec_type is not None:
            print(exec_type)
            print(exec_value)
            print(traceback)
        self.file_object.close()
