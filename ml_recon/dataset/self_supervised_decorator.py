import torch
import numpy as np
import random
from torch.utils.data import Dataset
from argparse import ArgumentParser

from typing import Union, Callable 
from ml_recon.dataset.undersample import gen_pdf_columns, calc_k, apply_undersampling, scale_pdf
from ml_recon.dataset.k_space_dataset import KSpaceDataset
from ml_recon.dataset.supervised_decorator import SupervisedDecorator

class SelfSupervisedDecorator(SupervisedDecorator):
    def __init__(
        self, 
        dataset: KSpaceDataset, 
        R: int = 8, 
        R_hat: int = 2,
        poly_order: int = 8,
        acs_lines: int = 10,
        transforms: Union[Callable, None] = None, 
        segregated: bool = False,
        line_constrained: bool = False,
    ):
        super().__init__(dataset, R, line_constrained, poly_order, acs_lines, transforms, segregated)
        self.R_hat = R_hat
        self.acs_lines = acs_lines
        self.random_index = random.randint(0, 10000)

        self.lambda_prob = scale_pdf(self.omega_prob, R_hat, acs_lines, line_constrained=line_constrained) 

        one_minus_eps = 1 - 1e-3
        self.lambda_prob[self.lambda_prob > one_minus_eps] = one_minus_eps

        self.k = torch.from_numpy(calc_k(self.lambda_prob, self.omega_prob)).float()
        self.transforms = transforms

    def __len__(self):
        return self.dataset.__len__()


    def __getitem__(self, index):
        k_space = self.dataset[index] #[con, chan, h, w] 
        
        under, mask_omega, new_prob = apply_undersampling(self.random_index + index, self.omega_prob, k_space, deterministic=True, line_constrained=self.line_constrained, segregated=self.segregated)
        # scale pdf
        scaled_new_prob = scale_pdf(new_prob, self.R_hat, self.acs_lines)
        doub_under, mask_lambda, _ = apply_undersampling(index, scaled_new_prob, under, deterministic=False, line_constrained=self.line_constrained, segregated=False)
        
        # loss mask is what is not in double undersampled
        target = under * ~mask_lambda
        
        output = {
                'input': doub_under, 
                'target': target, 
                'fs_k_space': k_space,
                'mask_omega': mask_omega, 
                'mask_lambda': mask_lambda
                }

        if self.transforms:
            output = self.transforms(output)
        return output

    @staticmethod
    def add_model_specific_args(parent_parser):  
        if parent_parser:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
        else:
            parser = ArgumentParser()

        parser.add_argument(
                "--R", 
                default=8,
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


