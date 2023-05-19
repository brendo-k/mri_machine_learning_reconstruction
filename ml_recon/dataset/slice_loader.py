from ml_recon.Dataset.FileReader.filereader import FileReader
from torch.utils.data import Dataset
import json
import os
from typing import (
    Callable,
    Optional,
    Union,
)


class SliceLoader(Dataset):
    """
    Takes data directory and creates a dataset. Before using you need to specify the file reader 
    to use in the filereader variable. 
    """
    filereader: FileReader
    def __init__(
            self, 
            index_info:Union[str, os.PathLike], 
            raw_sample_filter:Optional[Callable]=lambda _: True, # if not defined let everything though
            transforms:Optional[Callable]=None
            ):


        self.tranforms = transforms
        self.data_list = []
        with open(index_info, 'r') as f:
            self.index_info = json.load(f)
            for key in self.index_info:
                if raw_sample_filter(self.index_info[key]):
                    self.data_list.append(self.index_info[key])
    
    def set_file_reader(self, filereader: FileReader):
        self.filereader = filereader

    def __getitem__(self, index):
        slice_index = self.data_list[index]['slice_index']
        file_name = self.data_list[index]['file_name']
        images = self._read_files(file_name, slice_index)
        return images

    def __len__(self):
        return len(self.data_list)

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

if __name__ == '__main__':
    loader = SliceLoader('/home/kadotab/header.json', raw_sample_filter=lambda value: value['coils'] >=16)
    import cProfile
    from ml_recon.Dataset.FileReader.read_h5 import H5FileReader
    loader.set_file_reader(H5FileReader)
    cProfile.run('SliceLoader("/home/kadotab/header.json")')
    print(len(loader))

