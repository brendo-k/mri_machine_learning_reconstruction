import numpy as np
from numpy.typing import NDArray
import torch
import os
import math
from torch.utils.data import Dataset

from typing import Union, Callable
from ml_recon.utils.undersample_tools import (
    apply_undersampling_from_dist, 
    gen_pdf_columns,
    gen_pdf_bern, 
    ssdu_gaussian_selection
)

def get_unique_filename(base_name, extension=".png"):
    counter = 0
    filename = f"{base_name}{counter}{extension}"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_name}{counter}{extension}"
    return filename


PROBABILITY_DIST = os.environ.get('PROBABILITY_DIST', '')
class UndersampleDecorator(Dataset):
    """Decorator class that can be used on all datasets present in the dataset folder.
    The decorator wraps the original dataset and undersamples it based on parameters
    here. Can further do self supervised undersampling.

    """

    def __init__(
        self, 
        dataset, 
        R: float = 4, 
        poly_order: int = 8,
        acs_lines: int = 10,
        transforms: Union[Callable, None] = None, 
        self_supervised: bool = False, 
        R_hat: float = math.nan,
        original_ssdu_partioning: bool = False,
        sampling_method: str = '2d', 
        same_mask_every_epoch: bool = False,
        seed: Union[int, None] = 8,
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
        self.same_mask_every_epoch = same_mask_every_epoch
        self.self_supervised = self_supervised
        self.seed = seed

        # setting seeds for random masks
        rng = np.random.default_rng(seed)


        if self.sampling_type in ['2d', 'pi']:
            pdf_generator = gen_pdf_bern
        elif self.sampling_type == '1d':
            pdf_generator = gen_pdf_columns
        else:
            raise ValueError(f'Wrong sampling type! {self.sampling_type}')

        self.omega_prob = pdf_generator(dataset.nx, dataset.ny, 1/R, poly_order, acs_lines) 
        self.omega_prob = np.tile(self.omega_prob[np.newaxis, :, :], (self.contrasts, 1, 1))
        
        if PROBABILITY_DIST == '':
            self.lambda_prob = pdf_generator(dataset.nx, dataset.ny, 1/R_hat, poly_order, acs_lines) 
            self.lambda_prob = np.tile(self.lambda_prob[np.newaxis, :, :], (self.contrasts, 1, 1))
        else:
            self.lambda_prob = torch.load(PROBABILITY_DIST)

        if self.same_mask_every_epoch:
            self.lambda_seeds = np.random.default_rng(seed).integers(0, 2**32 - 1, size=len(dataset))
        else:
            self.lambda_seeds = None

        self.transforms = transforms


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        k_space:NDArray = self.dataset[index] #[con, chan, h, w] 
        fully_sampled_k_space = k_space.copy()

        zero_fill_mask = (fully_sampled_k_space != 0)
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
            input_mask, loss_mask = self.create_self_supervised_masks(first_undersampled, omega_mask, index)
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
                    'is_self_supervised': np.array([False])
                }
            )

        for keys in output:
            output[keys] = torch.from_numpy(output[keys])

        if self.transforms: 
            output = self.transforms(output)


        return output

    def create_self_supervised_masks(self, under, mask_omega, index):
        if self.original_ssdu_partioning:
            input_mask, loss_mask = ssdu_gaussian_selection(mask_omega)

        else:
            #seed = self.lambda_seeds[index].item()
            if self.lambda_seeds:
                # determenistic masks (same mask every epoch)
                _, mask_lambda = apply_undersampling_from_dist(self.lambda_seeds[index].item(), self.lambda_prob, under)
            else:
                _, mask_lambda = apply_undersampling_from_dist(None, self.lambda_prob, under)

            # loss mask is the disjoint set of the input mask
            input_mask = mask_omega * mask_lambda
            loss_mask = mask_omega * (1 - mask_lambda)

        return input_mask, loss_mask

    def compute_initial_mask(self, index, k_space):
        # same mask every time since the random seed is the index value
        if self.sampling_type in ['2d', '1d']: 
            under, mask_omega  = apply_undersampling_from_dist(
                index + self.seed,
                self.omega_prob,
                k_space, 
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
