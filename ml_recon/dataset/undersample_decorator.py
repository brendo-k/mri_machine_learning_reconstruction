import numpy as np
from numpy.typing import NDArray
import random
import torch
import math
from torch.utils.data import Dataset

from typing import Union, Callable
from ml_recon.utils.undersample_tools import apply_undersampling_from_dist, gen_pdf, scale_pdf, calc_k, ssdu_gaussian_selection

class UndersampleDecorator(Dataset):
    """Decorator class that can be used on all datasets present in the dataset folder.
    The decorator wraps the original dataset and undersamples it based on parameters
    here. Can further do self supervised undersampling.

    """

    def __init__(
        self, 
        dataset: Dataset, 
        R: float = 4, 
        R_hat: float = 2,
        polynomial_order: int = 8, # only used for variable density
        acs_lines: int = 10,
        initial_sampling_method: str = '2d', # can be 2d, 1d, pi
        is_ssdu_partitioning: bool = False, # if second mask is ssdu else use charlies method (same dist)
        transforms: Union[Callable, None] = None, 
    ):
        super().__init__()

        self.dataset = dataset
        self.contrasts = dataset[0].shape[0]
        self.contrast_order = dataset.contrast_order # type: ignore
        self.initial_sampling_method = initial_sampling_method
        self.R = R
        self.acs_lines = acs_lines
        self.is_ssdu_partitioning = is_ssdu_partitioning

        self.line_constrained = True if is_ssdu_partitioning == '1d' else False       

        if initial_sampling_method == '2d' or initial_sampling_method == '1d':
            self.omega_prob = gen_pdf(self.line_constrained, dataset.nx, dataset.ny, 1/R, polynomial_order, acs_lines) # type: ignore
            # create omega probability the same size as number of contrasts
            self.omega_prob = np.tile(self.omega_prob[np.newaxis, :, :], (self.contrasts, 1, 1))

        # create a random index for the determenistic seed
        self.random_index = random.randint(0, 1_000_000_000)
        self.acs_lines = acs_lines

        self.transforms = transforms

        self.R_hat = R_hat
        
    def __len__(self):
        return self.dataset.__len__() # type: ignore


    def __getitem__(self, index):
        k_space:NDArray[np.complex_] = self.dataset[index] #[con, chan, h, w] 
        if self.initial_sampling_method == '1d' or self.initial_sampling_method == '2d': 
            under, mask_omega  = apply_undersampling_from_dist(self.random_index + index, 
                                        self.omega_prob, 
                                        k_space, 
                                        deterministic=True, 
                                        line_constrained=self.line_constrained, 
                                        )
        elif self.initial_sampling_method == 'pi':
            mask_omega = self.gen_pi_mak(k_space)
            under = k_space * mask_omega
        else:
            raise ValueError('Could not find an inital sampling method')


        second_mask = self.gen_heuristic_ssl_masks(index, under, mask_omega)

        output = {
            'fs_k_space': k_space, 
            'undersampled': k_space * mask_omega,
            'initial_mask': mask_omega.astype(np.float32),
            'second_mask': second_mask.astype(np.float32)
        }
        
        for keys in output:
            output[keys] = torch.from_numpy(output[keys])

        if self.transforms: 
            output = self.transforms(output)

        return output

    def gen_pi_mak(self, k_space):
        mask_omega = np.zeros_like(k_space, dtype=bool)
        mask_omega[..., ::int(self.R)] = 1
        w = mask_omega.shape[-1]
        mask_omega[..., w//2-self.acs_lines//2:w//2+self.acs_lines//2] = 1
        return mask_omega

    def gen_heuristic_ssl_masks(self, index, under, mask_omega):
        if self.is_ssdu_partitioning:
            second_undersampling_mask = []
            loss_mask = []
            for i in range(mask_omega.shape[0]):
                input, loss = ssdu_gaussian_selection(mask_omega[i, 0])
                second_undersampling_mask.append(np.expand_dims(input, 0))
                loss_mask.append(np.expand_dims(loss, 0))

            second_undersampling_mask = np.stack(second_undersampling_mask, axis=0)
            loss_mask = np.stack(loss_mask, axis=0)
        else:
            scaled_new_prob = scale_pdf(self.omega_prob, self.R_hat, self.acs_lines)
            doub_under, second_undersampling_mask = apply_undersampling_from_dist(
                        index,
                        scaled_new_prob,
                        under,
                        deterministic=False,
                        line_constrained=self.line_constrained,
                        )
                
        
        return second_undersampling_mask 

