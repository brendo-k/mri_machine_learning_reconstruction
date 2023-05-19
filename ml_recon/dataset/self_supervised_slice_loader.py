from ml_recon.Dataset.undersampled_slice_loader import UndersampledSliceDataset
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
        unersampled_k_space = data['undersampled']

        # mask undersampled data agiain 
        double_undersampled, delta_mask = self.doubly_undersampled(unersampled_k_space, 4)
        data['double_undersample'] = double_undersampled
        data['delta_mask'] = delta_mask

        if self.transforms:
            data = self.transforms(data)
        return data
    
    def doubly_undersampled(self, k_space: torch.Tensor, R_hat: int):
        double_undersample = np.copy(k_space)
        indecies = super().get_undersampled_indecies(double_undersample, acs_width=self.acs_width, R=R_hat) 
        double_undersample = super().apply_undersampled_indecies(double_undersample, indecies)
        mask = super().build_mask(double_undersample, indecies)
        return double_undersample, mask

