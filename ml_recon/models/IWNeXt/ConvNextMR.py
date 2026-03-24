import torch
import torch.nn as nn
from ml_recon.models.IWNeXt.attention_ConvNext import Block
from ml_recon.models.IWNeXt.WaveletNet import WaveConvNeXt
from ml_recon.models.IWNeXt.wavelet_transform import IWT,DWT
from ml_recon.models.IWNeXt.mri_tools import rA, rfft2,rifft2
from ml_recon.utils import fft_2d_img, ifft_2d_img, complex_to_real, real_to_complex

class Dataconsistency(nn.Module):
    def __init__(self):
        super(Dataconsistency, self).__init__()
        self.lambda_dc = nn.Parameter(torch.tensor(1.0))
    def forward(self, x_rec: torch.Tensor, under_k: torch.Tensor, sub_mask:torch.Tensor, sense_maps: torch.Tensor):
        '''
        全部转变维度排列为(1,256,256,1) ???
        '''
        x_rec_cmplx = real_to_complex(x_rec)
        x_rec = sense_maps * x_rec_cmplx.unsqueeze(2)  # [B, contrast, C, H, W]
        k_rec= fft_2d_img(x_rec * (1.0 - sub_mask))
        k_out= (under_k + self.lambda_dc * k_rec) / (1 + self.lambda_dc)
        x_out = torch.sum(ifft_2d_img(k_out) * sense_maps.conj(), dim=2) / (sense_maps.conj() * sense_maps).sum(2)
        x_out = complex_to_real(x_out)
        return x_out

class ConvNet(nn.Module):
    def __init__(self, n_channels_in=2,n_channels_mid=32,n_channels_out=1):
        super(ConvNet, self).__init__()
        self.inConv=nn.Conv2d(n_channels_in,n_channels_mid,kernel_size=(3,3), padding=1)
        self.Blocks = []
        for i in range(4):
            self.Blocks.append(Block(dim=32))
        self.Blocks = nn.ModuleList(self.Blocks)
        self.sConv=nn.Conv2d(n_channels_mid,n_channels_mid,kernel_size=(3,3), padding=1)
        self.outConv=nn.Conv2d(n_channels_mid,n_channels_out,kernel_size=(3,3), padding=1)
    def forward(self,x):
        out_inia=self.inConv(x)
        x=out_inia
        for i in range(4):
            x=self.Blocks[i](x)
        x=self.sConv(x)
        x=self.outConv(x+out_inia)
        return x

class Generator(nn.Module):
    def __init__(self, input_channels):
        super(Generator, self).__init__()
        self.GC = ConvNet(input_channels,32,input_channels)
        self.wave = WaveConvNeXt(n_channels=input_channels*4,G0=32,kSize=3)
        self.DC=Dataconsistency()
        self.WDC=Dataconsistency()
        self.dwt = DWT()
        self.iwt=IWT()
        '''
        x:二次欠采样图像,是输入图像
        '''
    def forward(self,x,underimage,sub_mask, sense_maps):

        out=self.GC(x)
        i_out = self.DC(out, underimage, sub_mask, sense_maps)  # under_image是永恒不变的，x是前一个模块的输出

        x_tar_wave2=self.dwt(i_out)
        w_out=self.wave(x_tar_wave2)
        w_out=w_out+x_tar_wave2
        w_out=self.iwt(w_out)
        x_dc = self.WDC(w_out, underimage, sub_mask, sense_maps)
        
        x_dc = torch.clamp(x_dc, 0, 1)
        return  x_dc
