from torch.utils.data import Dataset
import h5py
import torch
import numpy as np
from ml_recon.utils.undersample_tools import gen_pdf, apply_undersampling

from dataset_output_type import TrainingSample

class ZeroShotDataset(Dataset):
    def __init__(
            self, 
            file_path, 
            validation=False, 
            is_undersampled=False, 
            R=4,
            R_hat=2,
            transforms = None
            ):
        self.file_path = file_path
        self.validation = validation
        self.is_undersampled = is_undersampled
        self.R = R
        self.R_hat = R_hat
        self.transforms = transforms

        with h5py.File(self.file_path) as fr:
            dataset = fr['kspace']
            assert dataset is h5py.Dataset
            self.slices = dataset.shape[0]
        
            

    def __len__(self):
        return self.slices

    def __getitem__(self, index):
        data = np.array([])
        with h5py.File(self.file_path) as fr:
            dataset = fr['kspace']
            assert dataset is h5py.Dataset
            data = dataset[:][index]
        # most likely shape is [contrast chan h w]

        chan, h, w = data.shape

        omega_prob = gen_pdf(False, w, h, 1/self.R, 8, 10)
        if self.is_undersampled:
            under = data
            mask_omega = under != 0
        else:
            under, mask_omega, _ = apply_undersampling(
                                    index, 
                                    omega_prob, 
                                    data, 
                                    deterministic=True, 
                                    line_constrained=False, 
                                    segregated=False
                                    )

        val_mask, training_mask = self.gen_validation_set(index, mask_omega)

        if self.validation:
            output = TrainingSample(
                input = training_mask*under,
                target = val_mask * under,
                fs_k_space = torch.from_numpy(data),
                mask = training_mask, 
                loss_mask = val_mask
            )

        else:
            lambda_prob = gen_pdf(False, w, h, 1/self.R_hat, 8, 10)
            _, mask_lambda, _ = apply_undersampling(
                                    index, 
                                    lambda_prob, 
                                    data, 
                                    deterministic=False, 
                                    line_constrained=False, 
                                    segregated=False
                                    )
            mask_lambda = mask_lambda * training_mask
            mask_loss = ~mask_lambda * training_mask
            output = TrainingSample(
                input = data * mask_lambda,
                target = data * mask_loss,
                fs_k_space = torch.from_numpy(data),
                mask = mask_lambda, 
                loss_mask = mask_loss
            )
        
        if self.transforms: 
            output = self.transforms(output)
        
        return output



    def gen_validation_set(self, index, mask_omega):
        rng = np.random.default_rng(index)
        mask_omega = mask_omega.astype(np.float32)
        mask_omega /= mask_omega.sum()
        
        # 20 percent of points should fall here
        mask_omega *= 0.2

        val_mask = (mask_omega - rng.uniform(0, 1, size=mask_omega.shape)) > 0 

        training_mask = mask_omega * ~val_mask

        return val_mask, training_mask
