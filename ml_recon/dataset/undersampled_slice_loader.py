import numpy as np
import os
from typing import Union, Callable

from ml_recon.dataset.filereader.read_h5 import H5FileReader
from ml_recon.dataset.slice_loader import SliceLoader 

class UndersampledSliceDataset(SliceLoader):
    """This is a supervised slice dataloader. 

    Args:
        meta_data (str, os.PathLike): path to metadata file holding slice information,
        acs_width (int): width of auto calibration lines to keep
        R (int): acceleration factor to use
    """
    def __init__(
            self,
            meta_data: Union[str, os.PathLike],
            acs_width: int = 10,
            R: int = 8,
            deterministic: bool = True,
            raw_sample_filter: Callable = lambda _: True,
            transforms: Callable = None
            ):

        # call super constructor
        super().__init__(meta_data, raw_sample_filter=raw_sample_filter)
        # define our file reader
        super().set_file_reader(H5FileReader)
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

    def get_item_from_index(self, index):
        data = super().__getitem__(index)

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
    

