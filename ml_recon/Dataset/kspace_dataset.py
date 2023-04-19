from .FileReader.filereader import FileReader
import os
from torch.utils.data import Dataset
import xmltodict
import numpy as np

class KSpaceDataset(Dataset):
    filereader: FileReader
    def __init__(self, directory, transforms=None):
        self.file_list = KSpaceDataset._get_files(directory)
        self.directory = directory
        self.tranforms = transforms

    def __getitem__(self, index):
        images = self._read_files(index)
        return images

    def __len__(self):
        return len(self.file_list)

    def _read_files(self, indexs):
        files = self.file_list[indexs]

        full_path = os.path.join(self.directory, files)
        reader = self.filereader(full_path)
        image = reader.read()
        k_space = image['kspace']
        ismrmrd_header = np.array(image['ismrmrd_header'], dtype=bytes)
        recon = image['reconstruction_rss']
        data = {
            'k_space': k_space[:], 
            'ismrmrd_header': xmltodict.parse(ismrmrd_header), 
            'recon': recon[:],
        }
        if self.tranforms:
            self.tranforms(data)
        reader.close()
        return data 

    def _get_files(directory):
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return files
