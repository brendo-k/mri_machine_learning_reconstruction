from torch.utils.data import Dataset
import h5py
import torch
import numpy as np
from ml_recon.utils.undersample_tools import gen_pdf, apply_undersampling

from ml_recon.dataset.dataset_output_type import TrainingSample
import torchvision.transforms.functional as F

class ZeroShotDataset(Dataset):
    def __init__(
            self, 
            file_path, 
            is_validation = False, 
            is_test = False,
            is_undersampled = False, 
            R = 4,
            R_hat = 2,
            transforms = None,
            nx = 256, 
            ny = 256,
            ):
        assert not is_test & is_validation, "is_validation and is_train both can't be set to true"

        self.file_path = file_path
        self.is_validation = is_validation
        self.is_test = is_test
        self.is_undersampled = is_undersampled
        self.R = R
        self.R_hat = R_hat
        self.transforms = transforms
        self.nx = nx
        self.ny = ny


        with h5py.File(self.file_path) as fr:
            dataset = fr['kspace']
            assert isinstance(dataset, h5py.Dataset)
            self.slices = dataset.shape[0]
        
            

    def __len__(self):
        return self.slices

    def __getitem__(self, index):
        with h5py.File(self.file_path) as fr:
            dataset = fr['kspace']
            assert isinstance(dataset, h5py.Dataset)
            data:np.ndarray = dataset[index]
        # most likely shape is [contrast chan h w]


        data = data[np.newaxis]
        k_space = self.resample_or_pad(torch.from_numpy(data), reduce_fov=True)
        _, _, h, w = k_space.shape

        omega_prob = gen_pdf(False, w, h, 1/self.R, 8, 10)
        omega_prob = omega_prob[np.newaxis]
        if self.is_undersampled:
            under = k_space
            mask_omega = under != 0
        else:
            under, mask_omega, _ = apply_undersampling(
                                    index, 
                                    omega_prob,
                                    k_space, 
                                    deterministic=True, 
                                    line_constrained=False, 
                                    segregated=False
                                    )
            under = under.numpy()
        
        # test case: pass all data and see how well ouput against fully sampled data
        if self.is_test:
            output = TrainingSample(
                input = torch.from_numpy(under),
                target = k_space,
                fs_k_space = k_space,
                mask = torch.from_numpy(mask_omega), 
                loss_mask = torch.ones_like(torch.from_numpy(under))
            )
            if self.transforms:
                output = self.transforms(output)
            return output


        val_mask, training_mask = self.gen_validation_set(index, mask_omega)

        # validation case: take a portion of undersampled data for validation
        if self.is_validation:
            output = TrainingSample(
                input = torch.from_numpy(training_mask * under),
                target = torch.from_numpy(val_mask * under),
                fs_k_space = k_space,
                mask = training_mask, 
                loss_mask = val_mask
            )

        # train case: sub-sample remaining set for training
        else:
            lambda_prob = gen_pdf(False, w, h, 1/self.R_hat, 8, 10)
            lambda_prob = lambda_prob[np.newaxis]
            _, mask_lambda, _ = apply_undersampling(
                                    index, 
                                    lambda_prob, 
                                    k_space, 
                                    deterministic=False, 
                                    line_constrained=False, 
                                    segregated=False
                                    )
            lambda_set = mask_lambda * training_mask
            loss_set = (~mask_lambda * training_mask)
            output = TrainingSample(
                input = torch.from_numpy(under * lambda_set),
                target = torch.from_numpy(under * loss_set),
                fs_k_space = k_space,
                mask = lambda_set, 
                loss_mask = loss_set
            )
        
        if self.transforms: 
            output = self.transforms(output)
        
        return output



    def gen_validation_set(self, index, mask_omega):
        rng = np.random.default_rng(index)
        mask_omega_prob = mask_omega.astype(np.float32)
        # 20 percent of points should fall here
        mask_omega_prob *= 0.2
        _, _, h, w = mask_omega_prob.shape
        mask_omega_prob[..., h//2-5:h//2+5, w//2-5:h//2+5] = 0
        
        val_mask = (mask_omega_prob - rng.uniform(0, 1, size=mask_omega.shape)) > 0 

        training_mask = mask_omega * ~val_mask

        return val_mask, training_mask


    def resample_or_pad(self, k_space, reduce_fov=True):
        """Takes k-space data and resamples data to desired height and width. If 
        the image is larger, we crop. If the image is smaller, we pad with zeros

        Args:
            k_space (np.ndarray): k_space to be cropped or padded 
            reduce_fov (bool, optional): If we should reduce fov along readout dimension. Defaults to True.

        Returns:
            np.ndarray: cropped k_space
        """
        resample_height = self.ny
        resample_width = self.nx
        if reduce_fov:
            k_space = k_space[..., ::2, :]

        return F.center_crop(k_space, [resample_height, resample_width])
