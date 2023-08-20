import nibabel as nib 

from .filereader import FileReader

class NiftiFileReader(FileReader):
    def __init__(self, file_name):
        self.file_name = file_name

    def __enter__(self):
        self.file_object = nib.load(self.file_name)
        return self.file_object.get_fdata()

    def get_keys(self):
        return self.file_object.keys()  

    def __exit__(self, exec_type, exec_value, traceback):
        if exec_type is not None:
            print(exec_type)
            print(exec_value)
            print(traceback)

