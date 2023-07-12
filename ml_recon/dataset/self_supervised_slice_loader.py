import random
import numpy as np
from typing import Optional, Callable

from ml_recon.dataset.undersampled_slice_loader import UndersampledSliceDataset

class SelfSupervisedSampling(UndersampledSliceDataset):
    def __init__(
            self, 
            h5_directory, 
            R, 
            R_hat, 
            transforms=None,
            raw_sample_filter: Optional[Callable] = lambda x: True
            ):

        # determanistic set to true so mask doesn't change every epoch
        super().__init__(h5_directory, R=R, raw_sample_filter=raw_sample_filter, deterministic=True)
        self.R_hat = R_hat
        self.transforms = transforms


    def __getitem__(self, index):
        
        data = super().get_item_from_index(index)

        # get undersampled k-space data
        undersampled = data['undersampled']

        # get probability density function of k-space along columns
        prob_lambda = super().gen_pdf_columns(undersampled.shape[-1], undersampled.shape[-2], 1/self.R_hat, 8, self.acs_width)
        
        # change mask every epoch
        omega_mask = super().get_mask_from_distribution(prob_lambda, np.random.default_rng())
        prob_omega = data['prob_omega']

        one_minus_eps = 1 - 1e-3
        prob_lambda[prob_lambda > one_minus_eps] = one_minus_eps

        data['double_undersample'] = undersampled * omega_mask
        data['omega_mask'] = omega_mask
        K = (1 - prob_omega) / (1 - prob_omega * prob_lambda)
        K = np.expand_dims(np.expand_dims(K, 0), 0).astype(float)
        data['K'] = K

        if self.transforms:
            data = self.transforms(data)
        return data