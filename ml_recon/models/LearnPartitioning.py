# ssl_model.py
import torch
import torch.nn as nn
from typing import Tuple, Literal
import numpy as np
from dataclasses import dataclass

from ml_recon.utils.kmax_relaxation import KMaxSoftmaxFunction
from ml_recon.utils.undersample_tools import gen_pdf_bern, gen_pdf_columns

@dataclass
class LearnPartitionConfig:
    image_size: Tuple[int, int, int]
    inital_R_value: float
    k_center_region: int = 10
    sigmoid_slope_probability: float = 5.0
    sigmoid_slope_sampling: float = 200
    is_warm_start: bool = True
    sampling_method: Literal['2d', '1d', 'pi'] = '2d'
    line_constrained: bool = False


class LearnPartitioning(nn.Module):
    """
    PyTorch module for learning partioning of k-space for self-supervised learning
    """
    def __init__(self, learn_part_config: LearnPartitionConfig):
        super().__init__()
        
        self.config = learn_part_config
        
        # Initialize partitioning weights W
        self._setup_sampling_weights(learn_part_config)


    def forward(self, initial_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nbatch = initial_mask.shape[0]

        lambda_set, inverse_set = self.split_into_lambda_loss_sets(initial_mask, nbatch)

        return lambda_set, inverse_set
    
    def split_into_lambda_loss_sets(self, omega_mask, batch_size): 
        lambda_mask = self.get_mask(batch_size)
        return omega_mask * lambda_mask, omega_mask * (1 - lambda_mask)

    def get_mask(self, batch_size):
        # Calculate probability and normalize
        norm_probability = self.get_probability_distribution()
        
        # make sure nothing is nan 
        assert not torch.isnan(norm_probability).any()

        # get random noise same size as the probability distribution for each item in the batch
        random_01_noise = torch.rand((batch_size,) + norm_probability.shape, device=norm_probability.device)

        activation = norm_probability - random_01_noise 
        # get sampling using softamx relaxation
        sampling_mask = self.kMaxSampling(activation, self.config.sigmoid_slope_sampling)
        
        #ensure sampling mask has no nans
        # Ensure sampling mask values are within [0, 1]
        assert not torch.isnan(sampling_mask).any()
        assert sampling_mask.min() >= 0 and sampling_mask.max() <= 1


        return sampling_mask.unsqueeze(2)
    
    
    def get_probability_distribution(self):
        sampling_weights = self.sampling_weights
        probability = torch.sigmoid(sampling_weights * self.config.sigmoid_slope_probability)
        # If this was LOUPE there would be a pdf normalization step here. We decide to omit this

        # set 10x10 box to always be sampled. Need to clone because of inplace opeartion
        c, h, w = sampling_weights.shape
        acs_prob = probability.clone()
        acs_prob[:, h//2-5:h//2+5,w//2-5:w//2+5] = 1

       
        return acs_prob
    
    def _setup_sampling_weights(self, config: LearnPartitionConfig):
        if config.is_warm_start: 
            if self.config.sampling_method == '2d':
                init_prob = gen_pdf_bern(config.image_size[1], config.image_size[2], 1/config.inital_R_value, 8, config.k_center_region).astype(np.float32)
            else: 
                #init_prob = gen_pdf_bern(config.image_size[1], config.image_size[2], 1/config.inital_R_value, 8, config.k_center_region).astype(np.float32)
                init_prob = gen_pdf_columns(config.image_size[1], config.image_size[2], 1/config.inital_R_value, 8, config.k_center_region).astype(np.float32)
            init_prob = torch.from_numpy(np.tile(init_prob[np.newaxis, :, :], (config.image_size[0], 1, 1)))
            init_prob = init_prob/(init_prob.max() + 2e-4) + 1e-4
        else:
            init_prob = torch.zeros(config.image_size) + 0.5
            h, w = init_prob.shape[1], init_prob.shape[2]
            init_prob[:, h//2 - config.k_center_region//2:h//2 + config.k_center_region//2, w//2 - config.k_center_region//2:w//2 + config.k_center_region//2] = 0.99
        self.sampling_weights = nn.Parameter(-torch.log((1/init_prob) - 1) / config.sigmoid_slope_probability)

    def get_R(self) -> torch.Tensor:
        probability = self.get_probability_distribution()
        cur_R = torch.ones(self.config.image_size[0], device=self.sampling_weights.device)
        for i in range(self.config.image_size[0]):
            cur_R[i] = 1/probability[i].mean()
                
        return cur_R

    def kMaxSampling(self, input, slope) -> torch.Tensor:
        return KMaxSoftmaxFunction.apply(input, slope) # type: ignore
