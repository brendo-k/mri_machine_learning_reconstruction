from ml_recon.Dataset.FileReader.filereader import FileReader
from torch.utils.data import Dataset
import json


class SliceLoader(Dataset):
    """
    Takes data directory and creates a dataset. Before using you need to specify the file reader 
    to use in the filereader variable. 
    """
    filereader: FileReader
    def __init__(self, index_info, transforms=None):
        self.tranforms = transforms
        with open(index_info, 'r') as f:
            self.index_info = json.load(f)
    
    def set_file_reader(self, filereader: FileReader):
        self.filereader = filereader

    def __getitem__(self, index):
        slice_index = self.index_info[str(index)]['slice_index']
        file_name = self.index_info[str(index)]['file_name']
        images = self._read_files(file_name, slice_index)
        return images

    def __len__(self):
        return len(self.index_info)

    def _read_files(self, file_name, slice_index):
        with self.filereader(file_name) as fr:
            slice = fr['kspace'][slice_index]
            recon_slice = fr['reconstruction_rss'][slice_index]
            data = {
                'k_space': slice, 
                'recon': recon_slice,
            }
            if self.tranforms:
                self.tranforms(data)
        return data 

