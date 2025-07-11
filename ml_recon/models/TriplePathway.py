# ssl_model.py
import torch.nn as nn
import os 
from ml_recon.models.MultiContrastVarNet import MultiContrastVarNet, VarnetConfig
from dataclasses import dataclass
import torch


@dataclass
class DualDomainConifg:
    is_pass_inverse: bool = False
    is_pass_original: bool = False 
    inverse_no_grad: bool = False
    original_no_grad: bool = False
    pass_all_lines: bool = False
    pass_through_size: int = 10
    seperate_models: int = False

print('FINAL_DC_STEP', os.getenv("FINAL_DC_STEP"))
FINAL_DC_STEP = bool(os.getenv("FINAL_DC_STEP", "True").lower() in ["true", "1", "yes", "y"])
print('FINAL_DC_STEP', FINAL_DC_STEP)

class TriplePathway(nn.Module):
    """
    Dual domain self-supervised learning module. Handles passing inverse, lambda, and original
    data through a single reconstruciton network.

    
    """
    def __init__(self, dual_domain_config: DualDomainConifg, varnet_config: VarnetConfig):
        super().__init__()
        self.dual_domain_config = dual_domain_config
        # same model used for each pathway
        self.recon_model = MultiContrastVarNet(varnet_config)

        if dual_domain_config.seperate_models: 
            self.inverse_model = MultiContrastVarNet(varnet_config)
            #self.full_model = MultiContrastVarNet(varnet_config)

    # undersampling mask can be: original undersampling mask or lambda set
    # loss mask can be all ones in the supervised case or the mask representing the inverse set
    def forward(self, undersampled_k, fully_sampled_k, input_set, target_set, return_all=False) -> dict:
        estimate_lambda = self.pass_through_lambda_path(undersampled_k, fully_sampled_k, input_set)

        estimate_inverse = None
        if self.dual_domain_config.is_pass_inverse or return_all:
            if self.dual_domain_config.inverse_no_grad:
                with torch.no_grad():
                    estimate_inverse = self.pass_through_inverse_path(undersampled_k, fully_sampled_k, input_set, target_set)
            else:
                estimate_inverse = self.pass_through_inverse_path(undersampled_k, fully_sampled_k, input_set, target_set)


        # these pathways only make sense in the self-supervised case, pass through original undersampled data
        estimate_full = None
        if self.dual_domain_config.is_pass_original or return_all:
            if self.dual_domain_config.original_no_grad:
                with torch.no_grad():
                    estimate_full = self.pass_through_model(self.recon_model, undersampled_k, input_set + target_set, fully_sampled_k)
            else:
                estimate_full = self.pass_through_model(self.recon_model, undersampled_k, input_set + target_set, fully_sampled_k)


        return {
            'full_path': estimate_full,
            'inverse_path': estimate_inverse,
            'lambda_path': estimate_lambda,
        }
    
    
    def pass_through_model(self, model, undersampled, mask, fully_sampled):
        # save some memory by not saving full image
        zero_pad_mask = fully_sampled[:, :, 0, :, :] != 0
        zero_pad_mask = zero_pad_mask.unsqueeze(2)

        estimate = model(undersampled*mask, mask, zero_pad_mask) # set zero regions to 1 in the mask
        if FINAL_DC_STEP:
            estimate = self.final_dc_step(undersampled, estimate, mask)
        return estimate * zero_pad_mask

    # replace estimated points with aquired points
    def final_dc_step(self, undersampled, estimated, mask):
        return estimated * (1 - mask) + undersampled * mask


    def pass_through_inverse_path(self, undersampled, fs_k_space, lambda_set, inverse_set):
        mask_inverse_w_acs, _ = TriplePathway.create_inverted_masks(
            lambda_set, 
            inverse_set, 
            self.dual_domain_config.pass_through_size, 
            self.dual_domain_config.pass_all_lines
        )
        
        if self.dual_domain_config.seperate_models:
            model = self.inverse_model
        else:
            model = self.recon_model
        estimate_inverse = self.pass_through_model(model, undersampled*mask_inverse_w_acs, mask_inverse_w_acs, fs_k_space)
            
        return estimate_inverse

    def pass_through_lambda_path(self, undersampled, fs_k_space, input_set):
        estimate_lambda = self.pass_through_model(self.recon_model, undersampled * input_set, input_set, fs_k_space)

        return estimate_lambda

    @staticmethod
    def create_inverted_masks(lambda_set, inverse_set, pass_through_size, pass_all_lines):
        _, _, _, h, w = lambda_set.shape
        mask_inverse_w_acs = inverse_set.clone()
        mask_lambda_wo_acs = lambda_set.clone()

        lower_bound = pass_through_size // 2
        upper_bound = pass_through_size - lower_bound
        if pass_all_lines:
            center_slice_h = slice(0, h)
        else:
            center_slice_h = slice(h//2-lower_bound, h//2+upper_bound)
        center_slice_w = slice(w//2-lower_bound, w//2+upper_bound)
        mask_inverse_w_acs[:, :, :, center_slice_h, center_slice_w] = lambda_set[:, :, :, center_slice_h, center_slice_w]
        mask_lambda_wo_acs[:, :, :, center_slice_h, center_slice_w] = inverse_set[:, :, :, center_slice_h, center_slice_w]
        return mask_inverse_w_acs, mask_lambda_wo_acs
