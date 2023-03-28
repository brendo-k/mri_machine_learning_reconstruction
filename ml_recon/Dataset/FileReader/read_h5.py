import h5py
from .filereader import FileReader

class H5FileReader(FileReader):
    def __init__(self, file_name):
        self.file_name = file_name
        self.file_object = h5py.File(self.file_name, 'r')

    def read(self):
        return self.file_object

    def get_keys(self):
        return self.file_object.keys()  

    def close(self):
        self.file_object.close()

