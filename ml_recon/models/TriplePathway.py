# ssl_model.py
import torch.nn as nn
from ml_recon.models.MultiContrastVarNet import MultiContrastVarNet, VarnetConfig
from dataclasses import dataclass

@dataclass
class DualDomainConifg:
    is_pass_inverse: bool = False
    is_pass_original: bool = False 

class TriplePathway(nn.Module):
    """
    Dual domain self-supervised learning module. Handles passing inverse, lambda, and original
    data through reconstruction networks.
    """
    def __init__(
            self,
            dual_domain_config: DualDomainConifg,
            varnet_config: VarnetConfig
        ):
        super().__init__()
        self.config = dual_domain_config
        self.recon_model = MultiContrastVarNet(varnet_config)

    # undersampling mask can be: original undersampling mask or lambda set
    # loss mask can be all ones in the supervised case or the mask representing the inverse set
    def forward(self, undersampled_k, fully_sampled_k, input_set, target_set, return_all=False):
        estimate_lambda = self.pass_through_lambda_path(undersampled_k, fully_sampled_k, input_set)

        # these pathways only make sense in the self-supervised case
        estimate_inverse = None
        if self.config.is_pass_inverse or return_all:
            # create new masks with inverted acs lines
            estimate_inverse = self.pass_through_inverse_path(undersampled_k, fully_sampled_k, undersampled_k, target_set)

        # these pathways only make sense in the self-supervised case, pass through original udnersampled data
        estimate_full = None
        if self.config.is_pass_original or return_all:
            estimate_full = self.pass_through_model(undersampled_k, input_set + target_set, fully_sampled_k)


        return {
            'full_path': estimate_full,
            'inverse_path': estimate_inverse,
            'lambda_path': estimate_lambda,
        }
    
    
    def pass_through_model(self, undersampled, mask, fully_sampled):
        zero_pad_mask = fully_sampled != 0
        estimate = self.recon_model(undersampled*mask, mask)
        estimate = self.final_dc_step(undersampled, estimate, mask)
        return estimate * zero_pad_mask

    def final_dc_step(self, undersampled, estimated, mask):
        return estimated * (1 - mask) + undersampled * mask


    def pass_through_inverse_path(self, undersampled, fs_k_space, lambda_set, inverse_set):
        mask_inverse_w_acs, _ = TriplePathway.create_inverted_masks(lambda_set, inverse_set)

        estimate_inverse = self.pass_through_model(undersampled*mask_inverse_w_acs, mask_inverse_w_acs, fs_k_space)
            
        return estimate_inverse

    @staticmethod
    def create_inverted_masks(lambda_set, inverse_set):
        _, _, _, h, w = lambda_set.shape
        mask_inverse_w_acs = inverse_set.clone()
        mask_lambda_wo_acs = lambda_set.clone()
        mask_inverse_w_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 1
        mask_lambda_wo_acs[:, :, :, h//2-5:h//2+5, w//2-5:w//2+5] = 0
        return mask_inverse_w_acs,mask_lambda_wo_acs

    def pass_through_lambda_path(self, undersampled, fs_k_space, lambda_set):
        estimate_lambda = self.pass_through_model(undersampled * lambda_set, lambda_set, fs_k_space)

        return estimate_lambda
    