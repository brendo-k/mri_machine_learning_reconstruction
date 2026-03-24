import torch.nn as nn
import torch 

# personal code
from ml_recon.models.IWNeXt.ConvNextMR import Generator
from ml_recon.models.IWNeXt.wavelet_transform import DWT as dwt
from ml_recon.models import SensetivityModel_mc
from ml_recon.utils import ifft_2d_img, fft_2d_img

class IWNeXt_mc(nn.Module):
    def __init__(self, rank, num_layers): 
        self.recon_network = ISTANetPlus(rank, num_layers)
        self.sense_model = SensetivityModel_mc(in_chans=2, out_chans=2, chans=8)

    def forward(self, k_space, sub_mask):
        if k_space.shape[1] <= 1:
            assert ValueError(f'Input needs at least 2 contrasts! Got {k_space.shape[1]}')
        sense_maps = self.sense_model(k_space, sub_mask)
        imgs = (ifft_2d_img(k_space) * sense_maps.conj()).sum(2) / (sense_maps.conj() * sense_maps).sum(2)
        outputs = []
        for contrast_index in range(imgs.shape[1]):
            contrast = imgs[:, contrast_index]
            mask_contrast = sub_mask[:, contrast_index]
            if outputs:
                comp_contrast = imgs[:, contrast_index+1]
            else:
                comp_contrast = outputs[0]
            outputs.append(
                self.recon_network(contrast, mask_contrast, comp_contrast)
            )

        output_imgs = torch.cat(outputs, dim=1)
        output_kspace = fft_2d_img(output_imgs) * sense_maps

        return output_kspace
            

class ISTANetPlus(nn.Module):
    def __init__(self,rank,num_layers):
        super(ISTANetPlus, self).__init__()
        self.rank=rank
        self.num_layers = num_layers
        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(Generator(self.rank))
        self.layers = nn.ModuleList(self.layers)
    def forward(self,input_img,sub_mask,PD_label):
        x = input_img
        for i in range(self.num_layers):
            x= self.layers[i](x,input_img,sub_mask,PD_label)
        x_final = x
        return x_final

class ParallelNetwork(nn.Module):
    def __init__(self,rank,num_layers):
        super(ParallelNetwork, self).__init__()
        self.rank=rank
        self.num_layers = num_layers
        self.network = ISTANetPlus(self.rank,self.num_layers)
        self.dwt = dwt()
    def forward(self, under_img_up,mask_up,under_img_down,mask_down,PD_label):
        output_up= self.network(under_img_up,mask_up,PD_label)
        output_down= self.network(under_img_down,mask_down,PD_label)
        output_up_wave=self.dwt(output_up)
        output_down_wave = self.dwt(output_down)
        return output_up,output_up_wave,output_down,output_down_wave
