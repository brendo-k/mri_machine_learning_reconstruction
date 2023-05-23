import random

from ml_recon.dataset.undersampled_slice_loader import UndersampledSliceDataset
import random
import numpy as np
import torch 

class SelfSupervisedSampling(UndersampledSliceDataset):
    def __init__(self, h5_directory, R, R_hat):
        super().__init__(h5_directory, R=R)
        self.R_hat = R_hat

    def __getitem__(self, index):
        data = super().__getitem__(index)

        # get undersampled k-space data
        undersampled = data['undersampled']

        prob_lambda = super().gen_pdf_columns(undersampled.shape[-2], undersampled.shape[-1], 1/self.R_hat, 8, self.acs_width)
        lambda_mask = super().mask_from_prob(prob_lambda)
        prob_omega = data['prob_omega']

        data['double_undersample'] = undersampled * lambda_mask
        data['lambda_mask'] = lambda_mask
        K = torch.as_tensor((1 - prob_omega) / (1 - prob_omega * prob_lambda))
        K = K.float().unsqueeze(0).unsqueeze(0)
        data['K'] = K

        if self.transforms:
            data = self.transforms(data)
        return data