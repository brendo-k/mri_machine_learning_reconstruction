import numpy as np
import random
import torch
import math
from torch.utils.data import Dataset

from typing import Union, Callable
from ml_recon.utils.undersample_tools import apply_undersampling, gen_pdf, scale_pdf, calc_k
from ml_recon.dataset.dataset_output_type import TrainingSample

class UndersampleDecorator(Dataset):
    """Decorator class that can be used on all datasets present in the dataset folder.
    The decorator wraps the original dataset and undersamples it based on parameters
    here. Can further do self supervised undersampling.

    """

    def __init__(
        self, 
        dataset: Dataset, 
        R: float = 4, 
        line_constrained: bool = True, 
        poly_order: int = 8,
        acs_lines: int = 10,
        transforms: Union[Callable, None] = None, 
        segregated: bool = False,
        self_supervised: bool = False, 
        R_hat: float = math.nan,
    ):
        super().__init__()

        self.dataset = dataset
        self.contrasts = dataset[0].shape[0]
        self.contrast_order = dataset.contrast_order # type: ignore
        self.line_constrained = line_constrained
        self.segregated = segregated

        self.omega_prob = gen_pdf(line_constrained, dataset.nx, dataset.ny, 1/R, poly_order, acs_lines) # type: ignore
        # create omega probability the same size as number of contrasts
        self.omega_prob = np.tile(self.omega_prob[np.newaxis, :, :], (self.contrasts, 1, 1))

        self.transforms = transforms
        self.random_index = random.randint(0, 1_000_000_000)
        self.acs_lines = acs_lines

        #self supervised
        self.self_supervised = self_supervised
        if self.self_supervised:
            assert not math.isnan(R_hat)
            self.R_hat = R_hat

            #second probability mask scaled to desired R_hat value
            self.lambda_prob = scale_pdf(self.omega_prob, R_hat, acs_lines, line_constrained=line_constrained) 

            one_minus_eps = 1 - 1e-3
            self.lambda_prob[self.lambda_prob > one_minus_eps] = one_minus_eps
            
            # scaling factor if using k-weighted ssdu
            self.k = torch.from_numpy(calc_k(self.lambda_prob, self.omega_prob)).float()


    def __len__(self):
        return self.dataset.__len__() # type: ignore


    def __getitem__(self, index):
        k_space = self.dataset[index] #[con, chan, h, w] 
        
        under, mask_omega, new_prob = apply_undersampling(self.random_index + index, 
                                       self.omega_prob, 
                                       k_space, 
                                       deterministic=True, 
                                       line_constrained=self.line_constrained, 
                                       segregated=self.segregated
                                       )
        output = TrainingSample(
                input = under, 
                target = k_space, 
                fs_k_space = k_space.clone(),
                mask = torch.from_numpy(mask_omega),
                loss_mask = torch.ones_like(torch.from_numpy(mask_omega))
                )

        if self.self_supervised:

            # scale pdf
            scaled_new_prob = scale_pdf(new_prob, self.R_hat, self.acs_lines)
            doub_under, mask_lambda, _ = apply_undersampling(
                    index,
                    scaled_new_prob,
                    under,
                    deterministic=False,
                    line_constrained=self.line_constrained,
                    segregated=False
                    )
            
            # loss mask is what is not in double undersampled
            target = under * ~mask_lambda
            output = TrainingSample(
                input = doub_under, 
                target = target, 
                fs_k_space = k_space,
                mask = torch.from_numpy(mask_lambda & mask_omega),
                loss_mask = torch.from_numpy(~mask_lambda & mask_omega)
                )

        if self.transforms: 
            output = self.transforms(output)

        return output

