import numpy as np
import random
from torch.utils.data import Dataset

from typing import Union, Callable
from ml_recon.dataset.undersample import apply_undersampling, gen_pdf
from ml_recon.dataset.k_space_dataset import KSpaceDataset

class SupervisedDecorator(Dataset):
    def __init__(
        self, 
        dataset: KSpaceDataset, 
        R: int = 4, 
        line_constrained: bool = True, 
        poly_order: int = 8,
        acs_lines: int = 10,
        transforms: Union[Callable, None] = None, 
    ):
        super().__init__()

        self.dataset = dataset
        self.contrasts = dataset[0].shape[0]
        self.contrast_order = dataset.contrast_order
        self.line_constrained = line_constrained

        self.omega_prob = gen_pdf(line_constrained, dataset.nx, dataset.ny, 1/R, poly_order, acs_lines)
        # create omega probability the same size as number of contrasts
        self.omega_prob = np.tile(self.omega_prob[np.newaxis, :, :], (self.contrasts, 1, 1))

        self.transforms = transforms
        self.random_index = random.randint(0, 1_000_000_000)

    def __len__(self):
        return self.dataset.__len__()


    def __getitem__(self, index):
        k_space = self.dataset[index] #[con, chan, h, w] 
        
        under, _ = apply_undersampling(self.random_index + index, self.omega_prob, k_space, deterministic=True, line_constrained=self.line_constrained)

        if self.transforms: 
            under, k_space = self.transforms((under, k_space))

        return under, k_space

