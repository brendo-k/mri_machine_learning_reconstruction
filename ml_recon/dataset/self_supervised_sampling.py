from ml_recon.Dataset.undersampled_dataset import UndersampledKSpaceDataset
import random
import numpy as np
import torch 

class SelfSupervisedSampling(UndersampledKSpaceDataset):
    def __init__(self, h5_directory, R, R_hat):
        super.__init__(h5_directory, R=R)
        self.R_hat = R_hat

    def __getitem__(self, index):
        data = super().__getitem__(index)

        # get k-space data
        undersampled = data['undersampled']

        # get double undersampled amount
        double_undersampled, delta_mask = self.doubly_undersampled(undersampled, self.R_hat)
        data['double_undersample'] = double_undersampled
        data['delta_mask'] = delta_mask

        # apply mask
        if self.transforms:
            data = self.transforms(data)
        return data
    
    def doubly_undersampled(k_space: torch.Tensor, R_hat: int):
        double_undersample = k_space.clone()
        indecies = super().get_undersampled_indecies(acs_width=10, R=R_hat) 
        double_undersample = super().apply_undersampled_indecies(double_undersample, indecies)
        mask = super().build_mask(double_undersample, indecies)
        return double_undersample, mask

