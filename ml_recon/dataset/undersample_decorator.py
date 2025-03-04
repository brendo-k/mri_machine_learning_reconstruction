import numpy as np
from numpy.typing import NDArray
import random
import torch
import math
from torch.utils.data import Dataset

from typing import Union, Callable
from ml_recon.utils.undersample_tools import (
    apply_undersampling_from_dist, 
    gen_pdf_columns, 
    gen_pdf_columns_charlie,
    gen_pdf_bern, 
    scale_pdf, 
    ssdu_gaussian_selection
    )


class UndersampleDecorator(Dataset):
    """Decorator class that can be used on all datasets present in the dataset folder.
    The decorator wraps the original dataset and undersamples it based on parameters
    here. Can further do self supervised undersampling.

    """

    def __init__(
        self, 
        dataset: Dataset, 
        R: float = 4, 
        poly_order: int = 8,
        acs_lines: int = 10,
        transforms: Union[Callable, None] = None, 
        self_supervised: bool = False, 
        R_hat: float = math.nan,
        original_ssdu_partioning: bool = False,
        sampling_method: str = '2d'
    ):
        super().__init__()

        self.dataset = dataset
        self.contrasts = dataset[0].shape[0]
        self.contrast_order = dataset.contrast_order # type: ignore
        self.sampling_type = sampling_method
        self.R = R
        self.R_hat = R_hat
        self.acs_lines = acs_lines
        self.original_ssdu_partioning = original_ssdu_partioning
        self.lambda_rng = np.random.default_rng() # generator for lambda mask seeds.
        print('hello!')
        
        assert (not self.original_ssdu_partioning or self_supervised), 'Only partioing if self-supervised!'

        if self.sampling_type == '2d' or self.sampling_type == 'pi':
            self.omega_prob = gen_pdf_bern(dataset.nx, dataset.ny, 1/R, poly_order, acs_lines) # type: ignore
            self.omega_prob = np.tile(self.omega_prob[np.newaxis, :, :], (self.contrasts, 1, 1))
            self.lambda_prob = gen_pdf_bern(dataset.nx, dataset.ny, 1/R_hat, poly_order, acs_lines) # type: ignore
            self.lambda_prob = np.tile(self.lambda_prob[np.newaxis, :, :], (self.contrasts, 1, 1))
        elif self.sampling_type == '1d':
            self.omega_prob = gen_pdf_columns_charlie(dataset.nx, dataset.ny, 1/R, poly_order, acs_lines) # type: ignore
            self.omega_prob = np.tile(self.omega_prob[np.newaxis, :, :], (self.contrasts, 1, 1))
            self.lambda_prob = gen_pdf_columns_charlie(dataset.nx, dataset.ny, 1/R_hat, poly_order, acs_lines) # type: ignore
            self.lambda_prob = np.tile(self.lambda_prob[np.newaxis, :, :], (self.contrasts, 1, 1))

        self.transforms = transforms
        self.acs_lines = acs_lines

        #self supervised
        self.self_supervised = self_supervised

    def __len__(self):
        return self.dataset.__len__() # type: ignore
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        seed = (self.lambda_rng.integers(0, 2**23) + self.epoch) % 2**23 
        self.lambda_rng = np.random.default_rng(seed)


    def __getitem__(self, index):
        k_space:NDArray = self.dataset[index] #[con, chan, h, w] 
        fully_sampled_k_space = k_space.copy()
        
        zero_fill_mask = fully_sampled_k_space != 0 
        first_undersampled, omega_mask = self.compute_initial_mask(index, k_space)
        omega_mask = omega_mask.astype(np.float32)
        
        # only mask where there is data. Exlcude zero filled data
        omega_mask = omega_mask * zero_fill_mask

        output = {
                'undersampled': first_undersampled, 
                'fs_k_space': fully_sampled_k_space,
                'mask': omega_mask,
                'loss_mask': np.ones_like(omega_mask)
                }

        if self.self_supervised:
            input_mask, loss_mask = self.create_self_supervised_masks(index, first_undersampled, omega_mask, output)
            output.update(
                {
                    'mask': input_mask.astype(np.float32),
                    'loss_mask': loss_mask.astype(np.float32),
                    'is_self_supervised': np.array([True])
                }
            )
        else:
            output.update(
                {
                    'is_self_supervised':np.array([False])
                }
            )
        
        for keys in output:
            output[keys] = torch.from_numpy(output[keys])

        if self.transforms: 
            output = self.transforms(output)

        return output

    def create_self_supervised_masks(self, index, under, mask_omega, output):
        if self.original_ssdu_partioning:
            input_mask = []
            loss_mask = []
            for i in range(mask_omega.shape[0]):
                input, loss = ssdu_gaussian_selection(mask_omega[i, 0])
                input_mask.append(np.expand_dims(input, 0))
                loss_mask.append(np.expand_dims(loss, 0))

            input_mask = np.stack(input_mask, axis=0)
            loss_mask = np.stack(loss_mask, axis=0)

        else:
            seed = self.lambda_rng.integers(0, 2**23)

            line_constrained = self.sampling_type == '1d'
            _, mask_lambda = apply_undersampling_from_dist(
                        seed,
                        self.lambda_prob,
                        under,
                        line_constrained=line_constrained,
                        )
                
            # loss mask is the disjoint set of the input mask
            input_mask = mask_omega * mask_lambda
            loss_mask = mask_omega * (1 - mask_lambda)

        return input_mask, loss_mask

    def compute_initial_mask(self, index, k_space):
        # same mask every time since the random seed is the index value
        if self.sampling_type == '2d' or self.sampling_type == '1d': 
            line_constrained = self.sampling_type == '1d'
            under, mask_omega  = apply_undersampling_from_dist(
                                        index,
                                        self.omega_prob,
                                        k_space, 
                                        line_constrained=line_constrained, 
                                        )
        elif self.sampling_type == 'pi':
            mask_omega = np.zeros_like(k_space, dtype=bool)
            mask_omega[..., ::int(self.R)] = 1
            w = mask_omega.shape[-1]
            mask_omega[..., w//2-self.acs_lines//2:w//2+self.acs_lines//2] = 1
            under = k_space * mask_omega
        else:
            raise ValueError(f'Could not load {self.sampling_type}')
        
        return under, mask_omega
