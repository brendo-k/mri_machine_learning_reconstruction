import torch
import numpy as np
import random
from torch.utils.data import Dataset
from argparse import ArgumentParser

from typing import Union, Callable 
from ml_recon.dataset.undersample import gen_pdf_columns, calc_k, apply_undersampling
from ml_recon.dataset.k_space_dataset import KSpaceDataset

class UndersampleDecorator(Dataset):
    def __init__(
        self, 
        dataset: KSpaceDataset, 
        R: int = 4, 
        R_hat: int = 2,
        poly_order: int = 8,
        acs_lines: int = 10,
        transforms: Union[Callable, None] = None
    ):
        super().__init__()

        self.dataset = dataset
        contrasts = dataset[0].shape[0]

        self.omega_prob = gen_pdf_columns(dataset.nx, dataset.ny, 1/R, poly_order, acs_lines)
        self.lambda_prob = gen_pdf_columns(dataset.nx, dataset.ny, 1/R_hat, poly_order, acs_lines)

        self.omega_prob = np.tile(self.omega_prob[np.newaxis, :, :], (contrasts, 1, 1))
        self.lambda_prob = np.tile(self.lambda_prob[np.newaxis, :, :], (contrasts, 1, 1))

        one_minus_eps = 1 - 1e-3
        self.lambda_prob[self.lambda_prob > one_minus_eps] = one_minus_eps
        self.random_index = random.randint(0, 10000)

        self.k = torch.from_numpy(calc_k(self.lambda_prob, self.omega_prob)).float()
        self.transforms = transforms

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        k_space = self.dataset[index] #[con, chan, h, w] OR [chan, h, w]
        
        under = apply_undersampling(self.random_index + index, self.omega_prob, k_space, deterministic=True)
        doub_under = apply_undersampling(index, self.lambda_prob, under, deterministic=False)

        data = (doub_under, under, k_space, self.k)
        if self.transforms:
            data = self.transforms(data)
        return data

    @staticmethod
    def add_model_specific_args(parent_parser):  
        if parent_parser:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
        else:
            parser = ArgumentParser()

        parser.add_argument(
                "--R", 
                default=4,
                type=int,
                help="Omega undersampling factor"
                )

        parser.add_argument(
                "--R_hat", 
                default=2,
                type=int,
                help="Lambda undersampling factor"
                )

        parser.add_argument(
                "--poly_order", 
                default=8,
                type=int,
                help="Polynomial order for undersampling"
                )

        parser.add_argument(
                "--acs_lines", 
                default=10,
                type=int,
                help="Number of lines to keep in auto calibration region"
                )

        return parser


