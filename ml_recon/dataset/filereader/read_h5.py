import h5py
from .filereader import FileReader

class H5FileReader(FileReader):
    def __init__(self, file_name):
        self.file_name = file_name

    def __enter__(self):
        self.file_object = h5py.File(self.file_name, 'r')
        return self.file_object

    def get_keys(self):
        return self.file_object.keys()  

    def __exit__(self, exec_type, exec_value, traceback):
        if exec_type is not None:
            print(exec_type)
            print(exec_value)
            print(traceback)
        self.file_object.close()

