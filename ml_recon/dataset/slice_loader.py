import json
import os
from typing import (
    Callable,
    Optional,
    Union,
)

from ml_recon.dataset.filereader.filereader import FileReader
from torch.utils.data import Dataset
from scipy.interpolate import interpn
import numpy as np


class SliceLoader(Dataset):
    """
    Takes data directory and creates a dataset. Before using you need to specify the file reader 
    to use in the filereader variable. 
    """
    filereader: FileReader


    def __init__(
            self,
            header_file: Union[str, os.PathLike], 
            raw_sample_filter: Optional[Callable] = lambda _: True,  # if not defined let everything though
            transforms: Optional[Callable] = None
            ):
        self.tranforms = transforms
        self.data_list = []
        with open(header_file, 'r') as f:
            self.index_info = json.load(f)
            for key in self.index_info:
                if raw_sample_filter(self.index_info[key]):
                    self.data_list.append(self.index_info[key])


    def set_file_reader(self, filereader: FileReader):
        self.filereader = filereader

    def __getitem__(self, index):
        images = self._get_data_from_index(index)
        images = self.resample(images)

        return images

    def __len__(self):
        return len(self.data_list)

    def _get_data_from_index(self, index):
        slice_index = self.data_list[index]['slice_index']
        file_name = self.data_list[index]['file_name']
        with self.filereader(file_name) as fr:
            slice = fr['kspace'][slice_index]
            recon_slice = fr['reconstruction_rss'][slice_index]
            data = {
                'k_space': slice, 
                'recon': recon_slice,
            }
        return data 
    
    def resample(self, data):
        resample_height = 128
        resample_width = 128
        k_space = data['k_space']
        k_space = k_space[:, ::2, :]
        _, height, width = k_space.shape
        height_cent, width_cent = int(height/2), int(width/2)
        k_space = k_space[:, height_cent - resample_height//2:height_cent + resample_height//2, width_cent - resample_width//2: width_cent + resample_width//2]
        data['k_space'] = k_space

        recon = data['recon']
        xv, yv = np.meshgrid(np.linspace(0, recon.shape[0]-1, 128), np.linspace(0, recon.shape[1]-1, 128))
        recon_downsampled = interpn((range(recon.shape[0]), range(recon.shape[1])), recon, (yv, xv), bounds_error=False)
        data['recon'] = recon_downsampled
        
        return data

if __name__ == '__main__':
    from ml_recon.dataset.filereader.read_h5 import H5FileReader
    from ml_recon.utils.image_slices import image_slices
    import matplotlib.pyplot as plt
    loader = SliceLoader('/home/kadotab/train.json')
    loader.set_file_reader(H5FileReader)
    data = loader[0]

    image_slices(data['k_space'], vmax=1e-5)
    plt.savefig('/home/kadotab/python/ml/resampled')
    plt.clf()
    plt.imshow(data['recon'])
    plt.savefig('/home/kadotab/python/ml/recon_resampled')
    