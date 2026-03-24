import torch.nn as nn
import torch 

# personal code
from ml_recon.models.IWNeXt.ConvNextMR import Generator
from ml_recon.models.IWNeXt.wavelet_transform import DWT as dwt
from ml_recon.models import SensetivityModel_mc
from ml_recon.utils import ifft_2d_img, fft_2d_img, complex_to_real, real_to_complex
            

class ISTANetPlus(nn.Module):
    def __init__(self,num_layers, contrasts):
        super(ISTANetPlus, self).__init__()
        self.sense_model = SensetivityModel_mc(
            2, 
            2, 
            chans=8, 
            upsample_method='conv', 
            )
        self.num_layers = num_layers
        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(Generator(input_channels=contrasts * 2))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, k_space, sub_mask, fully_sampled_k):
        zero_filled_mask = fully_sampled_k[:, :, [0], :, :] != 0
        # calculate sense maps
        sense_maps = self.sense_model(k_space, sub_mask)

        # conver to images
        under_img = (ifft_2d_img(k_space * sub_mask) * sense_maps.conj()).sum(2) 

        # convert to real for processing in network
        under_img = complex_to_real(under_img)
        
        # process through ISTANet with NeXt blocks and wavelet blocks
        x = under_img
        for i in range(self.num_layers):
            x = self.layers[i](x, k_space, sub_mask, sense_maps)
        
        # convert back to complex and k-space
        x_final = real_to_complex(x)
        k_final = fft_2d_img(x_final.unsqueeze(2) * sense_maps)
        k_final = k_final * zero_filled_mask
        return k_final
