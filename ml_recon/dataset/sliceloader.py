import numpy as np
import os
import json
from typing import Union, Callable
from torch.utils.data import Dataset
from scipy.interpolate import interpn

from ml_recon.dataset.filereader.read_h5 import H5FileReader
from ml_recon.dataset.filereader.filereader import FileReader
from ml_recon.utils.read_headers import make_header

class SliceDataset(Dataset):
    """This is a supervised slice dataloader. 

    Args:
        meta_data (str, os.PathLike): path to metadata file holding slice information,
        acs_width (int): width of auto calibration lines to keep
        R (int): acceleration factor to use
    """
    def __init__(
            self,
            data_dir: Union[str, os.PathLike],
            file_reader: FileReader = H5FileReader,
            raw_sample_filter: Callable = lambda _: True,
            transforms: Callable = None,
            build_new_header: bool = False
            ):

        # call super constructor
        super().__init__()

        self.data_list = []
        self.file_reader = file_reader

        header_file = os.path.join(data_dir, 'header.json')

        if not os.path.isfile(header_file) or build_new_header:
            print(f'Making header in {data_dir}')
            self.data_list = make_header(data_dir, self.file_reader, output=header_file, sample_filter=raw_sample_filter)
        else:    
            with open(header_file, 'r') as f:
                print('Header file found!')
                self.index_info = json.load(f)
                for index in self.index_info:
                    if raw_sample_filter(self.index_info[index]):
                        self.data_list.append(self.index_info[index])
        
        print(f'Found {len(self.data_list)} slices')

        # add transforms
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.get_data_from_file(index)
        data = self.resample_or_pad(data)
        assert data.shape == (16, 128, 128)

        if self.transforms:
            data = self.transforms(data)
        return data
    
    def get_data_from_file(self, index):
        slice_index = self.data_list[index]['slice_index']
        file_name = self.data_list[index]['file_name']
        with self.file_reader(file_name) as fr:
            k_space = np.ascontiguousarray(np.flip(fr['kspace'][slice_index], axis=1))
            assert not np.isnan(k_space).any()
            recon_slice = np.ascontiguousarray(np.flip(fr['recon'][slice_index], axis=1))
        return k_space 

    def resample_or_pad(self, k_space, reduce_fov=True):
        """Takes k-space data and resamples data to desired height and width. If 
        the image is larger, we crop. If the image is smaller, we pad with zeros

        Args:
            k_space (np.ndarray): k_space to be cropped or padded 
            reduce_fov (bool, optional): If we should reduce fov along readout dimension. Defaults to True.

        Returns:
            np.ndarray: cropped k_space
        """
        resample_height = 128
        resample_width = 128
        if reduce_fov:
            k_space = k_space[:, ::2, :]
        _, height, width = k_space.shape
        height_cent, width_cent = int(height/2), int(width/2)
        if height > resample_height:
            k_space = k_space[:, height_cent - resample_height//2:height_cent + resample_height//2, :]
        else:
            pad_top = (resample_height - height)//2
            pad_bottom = (resample_height - height)- pad_top
            k_space = np.pad(k_space, [pad_top, pad_bottom, 0, 0])

        if width > resample_width:
            k_space = k_space[:, :, width_cent - resample_width//2: width_cent + resample_width//2]
        else:
            pad_left = (resample_width - width)//2
            pad_right = (resample_width - width) - pad_top
            k_space = np.pad(k_space, [0, 0, pad_left, pad_right])
        return k_space


