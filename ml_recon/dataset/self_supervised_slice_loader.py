import random

from ml_recon.dataset.undersampled_slice_loader import UndersampledSliceDataset
import random
import numpy as np
import torch 

class SelfSupervisedSampling(UndersampledSliceDataset):
    def __init__(self, h5_directory, R, R_hat, transforms):
        super().__init__(h5_directory, R=R)
        self.R_hat = R_hat
        self.transforms = transforms

    def __getitem__(self, index):
        
        data = super().get_item_from_index(index)

        # get undersampled k-space data
        undersampled = data['undersampled']

        # get probability density function of k-space along columns
        prob_lambda = super().gen_pdf_columns(undersampled.shape[-2], undersampled.shape[-1], 1/self.R_hat, 8, 0)
        
        lambda_mask = super().mask_from_prob(prob_lambda)
        prob_omega = data['prob_omega']

        one_minus_eps = 1 - 1e-3
        prob_lambda[prob_lambda > one_minus_eps] = one_minus_eps

        data['double_undersample'] = undersampled * lambda_mask
        data['lambda_mask'] = lambda_mask
        K =(1 - prob_omega) / (1 - prob_omega * prob_lambda)
        K = np.expand_dims(np.expand_dims(K, 0), 0).astype(float)
        data['K'] = K

        if self.transforms:
            data = self.transforms(data)
        return data