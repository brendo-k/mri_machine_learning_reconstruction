# ssl_model.py
import torch.nn as nn
from ml_recon.models.MultiContrastVarNet import MultiContrastVarNet, VarnetConfig
from dataclasses import dataclass
import torch

@dataclass
class DualDomainConifg:
    is_pass_inverse: bool = False
    is_pass_original: bool = False 
    inverse_no_grad: bool = False
    original_no_grad: bool = False

class TriplePathway(nn.Module):
    """
    Dual domain self-supervised learning module. Handles passing inverse, lambda, and original
    data through reconstruction networks.
    """
    def __init__(
            self,
            dual_domain_config: DualDomainConifg, # config for triple pathway dual domain 
            varnet_config: VarnetConfig, # config for VarNet Reconstruction
            pass_through_size: int = 10 # size of center region passed through to inverse mask
        ):
        super().__init__()
        self.config = dual_domain_config
        self.recon_model = MultiContrastVarNet(varnet_config)
        self.pass_through_size = pass_through_size

    # undersampling mask can be: original undersampling mask or lambda set
    # loss mask can be all ones in the supervised case or the mask representing the inverse set
    def forward(self, undersampled_k, fully_sampled_k, input_set, target_set, return_all=False):
        estimate_lambda = self.pass_through_lambda_path(undersampled_k, fully_sampled_k, input_set)

        # these pathways only make sense in the self-supervised case
        estimate_inverse = None
        if self.config.is_pass_inverse or return_all:
            if self.config.inverse_no_grad:
                with torch.no_grad():
                    estimate_inverse = self.pass_through_inverse_path(undersampled_k, fully_sampled_k, input_set, target_set)
            else:
                estimate_inverse = self.pass_through_inverse_path(undersampled_k, fully_sampled_k, input_set, target_set)


        # these pathways only make sense in the self-supervised case, pass through original udnersampled data
        estimate_full = None
        if self.config.is_pass_original or return_all:
            if self.config.original_no_grad:
                with torch.no_grad():
                    estimate_full = self.pass_through_model(undersampled_k, input_set + target_set, fully_sampled_k)
            else:
                estimate_full = self.pass_through_model(undersampled_k, input_set + target_set, fully_sampled_k)


        return {
            'full_path': estimate_full,
            'inverse_path': estimate_inverse,
            'lambda_path': estimate_lambda,
        }
    
    
    def pass_through_model(self, undersampled, mask, fully_sampled):
        # save some memory by not saving full image
        zero_pad_mask = fully_sampled[:, :, 0, :, :] != 0
        zero_pad_mask = zero_pad_mask.unsqueeze(2)

        estimate = self.recon_model(undersampled*mask, mask)
        estimate = self.final_dc_step(undersampled, estimate, mask)
        return estimate * zero_pad_mask

    # replace estimated points with aquired points
    def final_dc_step(self, undersampled, estimated, mask):
        return estimated * (1 - mask) + undersampled * mask


    def pass_through_inverse_path(self, undersampled, fs_k_space, lambda_set, inverse_set):
        mask_inverse_w_acs, _ = TriplePathway.create_inverted_masks(lambda_set, inverse_set, self.pass_through_size)

        estimate_inverse = self.pass_through_model(undersampled*mask_inverse_w_acs, mask_inverse_w_acs, fs_k_space)
            
        return estimate_inverse

    def pass_through_lambda_path(self, undersampled, fs_k_space, input_set):
        estimate_lambda = self.pass_through_model(undersampled * input_set, input_set, fs_k_space)

        return estimate_lambda

    @staticmethod
    def create_inverted_masks(lambda_set, inverse_set, pass_through_size):
        _, _, _, h, w = lambda_set.shape
        mask_inverse_w_acs = inverse_set.clone()
        mask_lambda_wo_acs = lambda_set.clone()
        lower_bound = pass_through_size // 2
        upper_bound = pass_through_size - lower_bound
        center_slice = slice(h//2-lower_bound, h//2+upper_bound)
        mask_inverse_w_acs[:, :, :, center_slice, :] = lambda_set[:, :, :, center_slice, :]
        mask_lambda_wo_acs[:, :, :, center_slice, :] = inverse_set[:, :, :, center_slice, :]
        return mask_inverse_w_acs, mask_lambda_wo_acs
