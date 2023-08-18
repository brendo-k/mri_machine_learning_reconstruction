import numpy as np
import os
import json
from typing import Union, Callable
from torch.utils.data import Dataset
from scipy.interpolate import interpn

from ml_recon.dataset.filereader.read_h5 import H5FileReader
from ml_recon.dataset.filereader.filereader import FileReader
from ml_recon.utils.read_headers import make_header

class UndersampledSliceDataset(Dataset):
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
            acs_width: int = 10,
            R: int = 8,
            deterministic: bool = True,
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
                print(f'Found {len(self.index_info)} slices')
                for index in self.index_info:
                    if raw_sample_filter(self.index_info[index]):
                        self.data_list.append(self.index_info[index])

        # define our file reader
        self.acs_width = acs_width
        self.deterministic = deterministic
        # add transforms
        self.transforms = transforms
        self.R = R


    def __getitem__(self, index):
        
        data = self.get_item_from_index(index)

        if self.transforms:
            data = self.transforms(data)
        return data
    
    def __len__(self):
        return len(self.data_list)

    def get_item_from_index(self, index):
        data = self.get_data_from_file(index)
        data = self.resample(data)

        rng = self.get_random_generator(index=index)

        # get k-space data
        k_space = data['k_space']

        prob_map = self.gen_pdf_columns(k_space.shape[-1], k_space.shape[-2], 1/self.R, 8, self.acs_width)
        mask = self.get_mask_from_distribution(prob_map, rng)
        undersampled =  k_space * mask

        data['mask'] = mask
        data['undersampled'] = undersampled
        data['prob_omega'] = prob_map.copy()
        return data

    def get_random_generator(self, index=None):
        if self.deterministic:
            rng = np.random.default_rng(index)
        else:
            rng = np.random.default_rng()
        return rng

    def resample(self, data):
        resample_height = 320
        resample_width = 320
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

    def get_data_from_file(self, index):
        slice_index = self.data_list[index]['slice_index']
        file_name = self.data_list[index]['file_name']
        with self.file_reader(file_name) as fr:
            k_space = np.ascontiguousarray(np.flip(fr['kspace'][slice_index], axis=1))
            recon_slice = np.ascontiguousarray(np.flip(fr['recon'][slice_index], axis=1))
            data = {
                'k_space': k_space, 
                'recon': recon_slice,
            }
        return data 
            

    def gen_pdf_columns(self, nx, ny, one_over_R, poylnomial_power, c_sq):
    # generates 1D polynomial variable density with sampling factor delta, fully sampled central square c_sq
        xv = np.linspace(-1, 1, nx)
        r = np.abs(xv)
        # normalize to 1
        r /= np.max(r)
        prob_map = (1 - r) ** poylnomial_power
        prob_map[prob_map > 1] = 1
        prob_map[ny // 2 - c_sq // 2:nx // 2 + c_sq // 2] = 1

        a = -1
        b = 1
        eta = 1e-3
        ii = 1
        while 1:
            c = (a + b) / 2
            prob_map = (1 - r) ** poylnomial_power + c
            prob_map[prob_map > 1] = 1
            prob_map[prob_map < 0] = 0
            prob_map[nx // 2 - c_sq // 2:nx // 2 + c_sq // 2] = 1
            delta_current = np.mean(prob_map)
            if one_over_R > delta_current + eta:
                a = c
            elif one_over_R < delta_current - eta:
                b = c
            else:
                break
            ii += 1
            if ii == 100:
                break
        
        prob_map = np.tile(prob_map, (ny, 1))

        return prob_map
    
    def get_mask_from_distribution(self, prob_map, rng_generator):
        prob_map[prob_map > 0.99] = 1
        (nx, _) = np.shape(prob_map)
        mask1d = rng_generator.binomial(1, prob_map[0:1])
        mask = np.repeat(mask1d, nx, axis=0)
        return np.array(mask, dtype=bool)
    

