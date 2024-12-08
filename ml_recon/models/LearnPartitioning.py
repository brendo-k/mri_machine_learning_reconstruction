# ssl_model.py
import torch
import torch.nn as nn
import einops
from typing import List, Tuple, Optional
from ml_recon.utils.kmax_relaxation import KMaxSoftmaxFunction
from ml_recon.models.MultiContrastVarNet import MultiContrastVarNet, VarnetConfig
from ml_recon.utils import ifft_2d_img, root_sum_of_squares
from ml_recon.utils.undersample_tools import gen_pdf_bern
import numpy as np
from dataclasses import dataclass

@dataclass
class LearnPartitionConfig:
    image_size: Tuple[int, int, int]
    inital_R_value: float
    k_center_region: int = 10
    probability_sig_slope: float = 5.0
    sampling_sig_slope: float = 200
    is_learn_R: bool = False
    is_warm_start: bool = True

class LearnPartitioning(nn.Module):
    """
    PyTorch module for learning partioning of k-space for self-supervised learning
    """
    def __init__(
            self,
            learn_part_config: LearnPartitionConfig, 
        ):
        super().__init__()
        
        self.config = learn_part_config
        
        # Initialize R values
        self._setup_R_values(learn_part_config)
        self._setup_sampling_weights(learn_part_config)


    def forward(self, 
                undersampled_k: torch.Tensor, 
                initial_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        nbatch, contrast, coil, h, w = undersampled_k.shape

        lambda_set, inverse_set = self.split_into_lambda_loss_sets(initial_mask, nbatch)

        return lambda_set, inverse_set
    
    def split_into_lambda_loss_sets(self, omega_mask, batch_size): 
        lambda_mask = self.get_mask(batch_size, mask_center=True)
        return omega_mask * lambda_mask, omega_mask * (1 - lambda_mask)

    def get_mask(self, batch_size, mask_center=False):
        # Calculate probability and normalize
        norm_probability = self.get_probability(mask_center)
        
        # make sure nothing is nan 
        assert not torch.isnan(norm_probability).any()

        # get random noise same size as the probability distribution for each item in the batch
        random_01_noise = torch.rand((batch_size,) + norm_probability.shape, device=norm_probability.device)

        activation = norm_probability - random_01_noise 
        # get sampling using softamx relaxation
        sampling_mask = self.kMaxSampling(activation, self.config.sampling_sig_slope)
        
        #ensure sampling mask has no nans
        # Ensure sampling mask values are within [0, 1]
        assert not torch.isnan(sampling_mask).any()
        assert sampling_mask.min() >= 0 and sampling_mask.max() <= 1


        return sampling_mask.unsqueeze(2)
    
    
    def get_probability(self, mask_center):
        sampling_weights = self.sampling_weights

        # get probability from sampling weights. Pass through sigmoid. 
        # NOTE this list is needed because of errors in autograd otherwise. It is due to 
        # multiplying each probability map per contrast by a different scalar.
        probability = [torch.sigmoid(sampling_weight * self.config.probability_sig_slope) for sampling_weight in sampling_weights]

        # get an R value
        R_value = self.get_R()

        # normalize the prob distribution to this R (this step is why we need the list of tensors for probability distribution)
        norm_probability = self.norm_prob(probability, R_value, mask_center=mask_center)

        # we can now concat the list together now. We are safe
        norm_probability = torch.stack(norm_probability, dim=0)
        return norm_probability
    
    
    def norm_prob(self, probability:List[torch.Tensor], cur_R:List[torch.Tensor], center_region=10, mask_center=False):
        image_shape = probability[0].shape

        probability = self.norm_2d_probability(probability, cur_R, center_region, mask_center, image_shape)

        # testing function to ensure probabilities are close to the set R value
        for probs, R in zip(probability, cur_R):
            assert probs.mean().item() - 1/R < 1e-3
             
        return probability
    
    
    def norm_2d_probability(self, probability, cur_R, center_region, mask_center, image_shape):
        center = [image_shape[0]//2, image_shape[1]//2]

        center_bb_x = slice(center[0]-center_region//2,center[0]+center_region//2)
        center_bb_y = slice(center[1]-center_region//2,center[1]+center_region//2)
            
        probability_sum = torch.zeros((len(probability), 1), device=probability[0].device)

        # create acs mask of zeros for acs box and zeros elsewhere
        center_mask = torch.ones(image_shape, device=probability[0].device)
        if mask_center:
            center_mask[center_bb_y, center_bb_x] = 0

        for i in range(len(probability)):
            probability[i] = probability[i] * center_mask
            probability_sum[i] = probability[i].sum(dim=[-1, -2])
            
        for i in range(len(probability)):
            probability_total = image_shape[-1] * image_shape[-2]/ cur_R[i]
            if mask_center:
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
                if mask_center:
                    inverse_sum -= center_region**2
                inverse_total = torch.maximum(inverse_total, torch.zeros_like(inverse_sum))
                scaling_factor = inverse_total / inverse_sum
                assert scaling_factor <= 1 and scaling_factor >= 0

                inv_prob = (1 - probability[i])*scaling_factor
                probability[i] = 1 - inv_prob
           
            # acs box is now ones and everything else is zeros
        if mask_center:
            for i in range(len(probability)):
                probability[i][center_bb_y, center_bb_x] = 1
        return probability
    
    def _setup_sampling_weights(self, config: LearnPartitionConfig):
        if config.is_warm_start: 
            init_prob = gen_pdf_bern(config.image_size[1], config.image_size[2], 1/config.inital_R_value, 8, config.k_center_region).astype(np.float32)
            init_prob = torch.from_numpy(np.tile(init_prob[np.newaxis, :, :], (config.image_size[0], 1, 1)))
            init_prob = init_prob/(init_prob.max() + 2e-4) + 1e-4
        else:
            init_prob = torch.zeros(config.image_size) + 0.5
        self.sampling_weights = nn.Parameter(-torch.log((1/init_prob) - 1) / config.probability_sig_slope)

    def _setup_R_values(self, config: LearnPartitionConfig):
        if config.is_learn_R: 
            self.learned_R_value = torch.Tensor(torch.full((config.image_size[0],), float(config.inital_R_value - 1)))
        else: 
            self.learned_R_value = torch.full((config.image_size[0],), float(config.inital_R_value))
            
    def get_R(self) -> List[torch.Tensor]:
        if self.config.is_learn_R:
            cur_R = []
            for R_val in self.learned_R_value:
                # no normalization of R, just ensure doesn't go less than 1
                cur_R.append(1 + torch.nn.functional.softplus(R_val))
                
        else: 
            cur_R = self.learned_R_value.tolist()
                
        return cur_R

    def kMaxSampling(self, input, slope) -> torch.Tensor:
        return KMaxSoftmaxFunction.apply(input, slope) # type: ignore