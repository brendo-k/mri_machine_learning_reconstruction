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
    sigmoid_slope_sampling: float = 200.0
    is_warm_start: bool = True
    sampling_method: Literal['2d', '1d', 'pi'] = '2d'
    is_learn_R: bool = True
    line_constrained: bool = False


class LearnPartitioning(nn.Module):
    """
    PyTorch module for learning partioning of k-space for self-supervised learning
    """
    def __init__(self, learn_part_config: LearnPartitionConfig):
        super().__init__()
        
        self.config = learn_part_config
        
        # Initialize partitioning weights W
        self._setup_R_values(learn_part_config)
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
        partitioning_probability = self.get_probability_distribution()
        
        # make sure nothing is nan 
        assert not torch.isnan(partitioning_probability).any()

        # get random noise same size as the probability distribution for each item in the batch
        random_01_noise = torch.rand((batch_size,) + partitioning_probability.shape, device=partitioning_probability.device)

        activation = partitioning_probability - random_01_noise 
        # get sampling using softamx relaxation
        sampling_mask = self.kMaxSampling(activation, self.config.sigmoid_slope_sampling)
        
        #ensure sampling mask has no nans
        # Ensure sampling mask values are within [0, 1]
        assert not torch.isnan(sampling_mask).any()
        assert sampling_mask.min() >= 0 and sampling_mask.max() <= 1


        return sampling_mask.unsqueeze(2)
    
    
    def get_probability_distribution(self):
        # unconstrained weights
        sampling_weights = self.sampling_weights
        # 0-1 bounding
        probability = [torch.sigmoid(weight * self.config.sigmoid_slope_probability) for weight in sampling_weights]

        # If this was LOUPE there would be a pdf normalization step here. We decide to omit this so LOUPE learns the R value
        probability = self.norm_prob(probability)

        return torch.stack(probability)
    

    def norm_prob(self, probability, center_region=10):
        image_shape = probability[0].shape
        cur_R = self.get_R()

        # if not learn probability, no need to norm
        probability = self.norm_2d_probability(probability, cur_R, center_region, image_shape)


        # testing function to ensure probabilities are close to the set R value
        for probs, R in zip(probability, cur_R):
            assert probs.mean().item() - 1/R < 1e-3
             
        return probability
    
    
    def norm_2d_probability(self, probability, cur_R, center_region, image_shape):
        center = [image_shape[0]//2, image_shape[1]//2]

        center_bb_x = slice(center[0]-center_region//2,center[0]+center_region//2)
        center_bb_y = slice(center[1]-center_region//2,center[1]+center_region//2)
            
        probability_sum = torch.zeros((len(probability), 1), device=probability[0].device)

        # create acs mask of zeros for acs box and zeros elsewhere
        center_mask = torch.ones(image_shape, device=probability[0].device)
        center_mask[center_bb_y, center_bb_x] = 0

        for i in range(len(probability)):
            probability[i] = probability[i] * center_mask
            probability_sum[i] = probability[i].sum(dim=[-1, -2])
            
        for i in range(len(probability)):
            probability_total = image_shape[-1] * image_shape[-2]/ cur_R[i]
            probability_total -= center_region ** 2

            # we need to find cur_R * scaling_factor = R
            # scaling down the values of 1
            if probability_sum[i] > probability_total:
                scaling_factor = probability_total / probability_sum[i]
                assert scaling_factor <= 1 and scaling_factor >= 0

                probability[i] = (probability[i] * scaling_factor)

            # scaling down the complement probability (scaling down 0)
            else:
                inverse_total = image_shape[1]*image_shape[0]*(1 - 1/cur_R[i])
                inverse_sum = (image_shape[1]*image_shape[0]) - probability_sum[i] 
                inverse_sum -= center_region**2
                inverse_total = torch.maximum(inverse_total, torch.zeros_like(inverse_sum))
                scaling_factor = inverse_total / inverse_sum
                assert scaling_factor <= 1 and scaling_factor >= 0

                inv_prob = (1 - probability[i])*scaling_factor
                probability[i] = 1 - inv_prob
                    
        # acs box is now ones and everything else is zeros
        for i in range(len(probability)):
            probability[i][center_bb_y, center_bb_x] = 1
           
        return probability


    def _setup_sampling_weights(self, config: LearnPartitionConfig):
        cur_R = self.get_R()
        if config.is_warm_start: 
            # initalize starting probability distribution
            if self.config.sampling_method in ['2d', 'pi']:
                init_prob = gen_pdf_bern(config.image_size[1], config.image_size[2], 1/cur_R[0], 8, config.k_center_region).astype(np.float32)
            else: 
                #init_prob = gen_pdf_bern(config.image_size[1], config.image_size[2], 1/config.inital_R_value, 8, config.k_center_region).astype(np.float32)
                init_prob = gen_pdf_columns(config.image_size[1], config.image_size[2], 1/cur_R[0], 8, config.k_center_region).astype(np.float32)

            # add contrast dims
            # ensure bounded by [1e-4 1-1e-4] to ensure no sigmoid inf
            init_prob = init_prob/(init_prob.max() + 2e-4) + 1e-4
        else:
            # init probability is all 0.5 (equal probability) 
            init_prob = np.zeros(config.image_size[1:]) + 0.5
            h, w = init_prob.shape[0], init_prob.shape[1]
            # set center distribution to be very high
            init_prob[h//2 - config.k_center_region//2:h//2 + config.k_center_region//2, w//2 - config.k_center_region//2:w//2 + config.k_center_region//2] = 0.99
        
        init_prob = torch.from_numpy(init_prob)
        # convert probability to sampling weights (inverse of sigmoid)
        sampling_weights = -torch.log((1/init_prob) - 1) / config.sigmoid_slope_probability

        # one sampling weight dist per contrast
        self.sampling_weights = nn.ParameterList([sampling_weights for _ in range(config.image_size[0])])

    def get_R(self):
        if self.config.is_learn_R:
            # if we are learning R, return the learned R value
            cur_R = 1 / torch.sigmoid(self.acceleration_rate)
        else:
            cur_R = 1 / self.acceleration_rate
        return cur_R


    def kMaxSampling(self, input, slope) -> torch.Tensor:
        return KMaxSoftmaxFunction.apply(input, slope) # type: ignore

    def _setup_R_values(self, config: LearnPartitionConfig):
        # Initalize the acceleration rate ( 1/R)
        if config.is_learn_R: 
            self.acceleration_rate = nn.Parameter(torch.logit(torch.full((config.image_size[0],), float(1/config.inital_R_value))))
        else: 
            self.acceleration_rate = torch.full((config.image_size[0],), float(1/config.inital_R_value))
