# ssl_model.py
import torch
import torch.nn as nn
import einops
from typing import List, Tuple, Optional
from ml_recon.utils.kmax_relaxation import KMaxSoftmaxFunction
from ml_recon.models import VarNet_mc
from ml_recon.utils import ifft_2d_img, root_sum_of_squares

class SSLModel(nn.Module):
    """
    PyTorch module for Self-Supervised Learning reconstruction model.
    Handles the core model logic and mask generation.
    """
    def __init__(
            self,
            image_size: Tuple[int, int, int],
            learned_R: float,
            channels: int = 32,
            center_region: int = 10,
            line_constrained: bool = False,
            sigmoid_slope_probability: float = 5.0,
            sigmoid_slope_sampling: float = 200,
            learn_R: bool = False,
            learn_sampling: bool = True, 
            pass_inverse_data: bool = False, 
            pass_original_data: bool = False,
        ):
        super().__init__()
        
        self.image_size = image_size
        self.center_region = center_region
        self.line_constrained = line_constrained
        self.sigmoid_slope_probability = sigmoid_slope_probability
        self.sigmoid_slope_sampling = sigmoid_slope_sampling
        self.learn_R = learn_R
        self.pass_inverse_data = pass_inverse_data
        self.pass_original_data = pass_original_data
        
        # Initialize reconstruction model
        self.recon_model = VarNet_mc(contrasts=image_size[0], chans=channels)
        
        # Initialize R values
        if learn_R:
            self.R_value = nn.Parameter(torch.full((image_size[0],), float(learned_R)))
        else:
            self.register_buffer('R_value', torch.full((image_size[0],), float(learned_R)))
        
        # Initialize sampling weights
        init_prob = torch.zeros(image_size) + 0.5
        self.sampling_weights = nn.Parameter(
            -torch.log((1/init_prob) - 1) / sigmoid_slope_probability,
            requires_grad=learn_sampling
        )

    def forward(self, undersampled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the reconstruction model.
        
        Args:
            undersampled: Undersampled k-space data
            mask: Sampling mask
            
        Returns:
            Reconstructed k-space data
        """
        initial_mask = (undersampled != 0).to(torch.float32)
        nbatch  = undersampled.shape[0]

        lambda_set, inverse_set = self.split_into_lambda_loss_sets(initial_mask, nbatch)

        estimate_lambda = self.pass_through_recon_model(undersampled, lambda_set)

        if self.pass_inverse_data:
            lambda_acs, inverse_acs = self.swap_acs_lines(lambda_set, inverse_set)
            estimate_inverse = self.pass_through_recon_model(undersampled, inverse_acs)

        if self.pass_original_data:
            estimate_original = self.pass_through_recon_model(undersampled, initial_mask)
    
    def split_into_lambda_loss_sets(self, omega_mask, batch_size): 
        lambda_mask = self.get_mask(batch_size, mask_center=True)
        return omega_mask * lambda_mask, omega_mask * (1 - lambda_mask)

    def get_mask(self, batch_size: int, mask_center: bool = False) -> torch.Tensor:
        """Generate sampling mask based on current parameters."""
        R_value, norm_probability = self._get_probability(mask_center)
        
        activation = norm_probability - torch.rand(
            (batch_size,) + self.image_size,
            device=self.sampling_weights.device
        )
    
        sampling_mask = self._kmax_sampling(activation)
        return sampling_mask.unsqueeze(2)

    def swap_acs_lines(self, lambda_set, inverse_set):
        nbatch, ncontrast, ncoil, nheight, nwidth = lambda_set.shape
        mask_inverse_w_acs = inverse_set.clone()
        mask_lambda_wo_acs = lambda_set.clone()
        mask_inverse_w_acs[:, :, :, nheight//2-5:nheight//2+5, nwidth//2-5:nwidth//2+5] = 1
        mask_lambda_wo_acs[:, :, :, nheight//2-5:nheight//2+5, nwidth//2-5:nwidth//2+5] = 0
        return mask_lambda_wo_acs, mask_inverse_w_acs
    
    def pass_through_recon_model(self, undersampled, mask):
        estimate = self.recon_model(undersampled * mask, mask)
        return self._final_dc_step(undersampled, estimate, mask)

    
    def _final_dc_step(self, undersampled: torch.Tensor, estimated: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Data consistency step."""
        return estimated * (1 - mask) + undersampled * mask


    
    def _get_probability(self, mask_center: bool) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Calculate sampling probabilities."""
        probability = [torch.sigmoid(w * self.sigmoid_slope_probability) for w in [self.sampling_weights]]
        R_value = self._norm_R(self.R_value)
        norm_probability = self._norm_prob(probability, R_value, mask_center=mask_center)
        return R_value, torch.stack(norm_probability, dim=0)
    
    def _kmax_sampling(self, input: torch.Tensor) -> torch.Tensor:
        """Apply passthrough relaxation."""
        return KMaxSoftmaxFunction.apply(input, self.sigmoid_slope_sampling)

    def _norm_R(self, R: torch.Tensor) -> List[torch.Tensor]:
        """Normalize R values."""
        if self.learn_R:
            return [1 + torch.nn.functional.softplus(R_val) for R_val in [R]]
        return [R]

    def _norm_prob(
            self,
            probability: List[torch.Tensor],
            cur_R: List[torch.Tensor],
            center_region: int = 10,
            mask_center: bool = False
        ) -> List[torch.Tensor]:
        """Normalize probabilities."""
        if self.line_constrained:
            return self._norm_1d_probability(probability, cur_R, center_region, mask_center)
        return self._norm_2d_probability(probability, cur_R, center_region, mask_center)
    
    def _norm_1d_probability(
            self,
            probability: List[torch.Tensor],
            cur_R: List[torch.Tensor],
            center_region: int,
            mask_center: bool
        ) -> List[torch.Tensor]:
        """Normalize 1D probabilities."""
        center = self.image_size[1] // 2
        center_bb_x = [center - self.center_region // 2, center + self.center_region // 2]
        
        center_mask = torch.ones(self.image_size[2], device=self.sampling_weights.device)
        if mask_center:
            center_mask[center_bb_x[0]:center_bb_x[1]] = 0
            
        normalized_probs = []
        for prob, R in zip(probability, cur_R):
            prob = prob * center_mask
            prob_sum = prob.sum(dim=-1) - prob[center_bb_x[0]:center_bb_x[1]].sum()
            
            prob_total = self.image_size[1] / R
            if mask_center:
                prob_total -= center_region
            prob_total = torch.maximum(prob_total, torch.zeros_like(prob_total))
            
            if prob_sum > prob_total:
                prob = prob * (prob_total / prob_sum)
            else:
                inverse_total = self.image_size[1] * (1 - 1/R)
                inverse_sum = self.image_size[1] - prob_sum
                if mask_center:
                    inverse_sum -= center_region
                scaling = inverse_total / inverse_sum
                prob = 1 - (1 - prob) * scaling
                
            if mask_center:
                prob = prob + (1 - center_mask)
            normalized_probs.append(prob)
            
        return normalized_probs

    def _norm_2d_probability(
            self,
            probability: List[torch.Tensor],
            cur_R: List[torch.Tensor],
            center_region: int,
            mask_center: bool
        ) -> List[torch.Tensor]:
        """Normalize 2D probabilities."""
        image_shape = probability[0].shape
        center = [image_shape[0]//2, image_shape[1]//2]
        center_bb = (
            slice(center[0]-center_region//2, center[0]+center_region//2),
            slice(center[1]-center_region//2, center[1]+center_region//2)
        )
        
        center_mask = torch.ones(image_shape, device=self.sampling_weights.device)
        if mask_center:
            center_mask[center_bb[0], center_bb[1]] = 0
            
        normalized_probs = []
        for prob, R in zip(probability, cur_R):
            prob = prob * center_mask
            prob_sum = prob.sum(dim=[-1, -2])
            
            prob_total = image_shape[-1] * image_shape[-2] / R
            if mask_center:
                prob_total -= center_region ** 2
                
            if prob_sum > prob_total:
                scaling = prob_total / prob_sum
                prob = prob * scaling
            else:
                inverse_total = image_shape[1] * image_shape[0] * (1 - 1/R)
                inverse_sum = (image_shape[1] * image_shape[0]) - prob_sum
                if mask_center:
                    inverse_sum -= center_region**2
                inverse_total = torch.maximum(inverse_total, torch.zeros_like(inverse_sum))
                scaling = inverse_total / inverse_sum
                prob = 1 - (1 - prob) * scaling
                
            if mask_center:
                prob[center_bb[0], center_bb[1]] = 1
            normalized_probs.append(prob)
            
        return normalized_probs