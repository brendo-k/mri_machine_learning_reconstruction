import torch
import torch.nn as nn
from mri_tools import rA, rfft2,rifft2
from net.attention_ConvNext import Block
from wavelet_transform import IWT,DWT
from net.WaveletNet import WaveConvNeXt

class Dataconsistency(nn.Module):
    def __init__(self):
        super(Dataconsistency, self).__init__()
    def forward(self,x_rec,under_img,sub_mask):
        '''
        全部转变维度排列为(1,256,256,1)
        '''
        under_img=under_img.permute(0, 2, 3, 1).contiguous()
        x_rec = x_rec.permute(0, 2, 3, 1).contiguous()
        under_k= rA(x_rec,(1.0 - sub_mask))
        x_k=rfft2(under_img)#输入图像的傅里叶变换
        k_out=x_k+under_k
        x_out=rifft2(k_out)
        x_out=x_out.permute(0, 3, 1, 2).contiguous()
        return x_out

class ConvNet(nn.Module):
    def __init__(self, rank,n_channels_in=2,n_channels_mid=32,n_channels_out=1):
        super(ConvNet, self).__init__()
        self.inConv=nn.Conv2d(n_channels_in,n_channels_mid,kernel_size=(3,3), padding=1)
        self.rank=rank
        self.Blocks = []
        for i in range(4):
            self.Blocks.append(Block(dim=32,rank=self.rank))
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
    def __init__(self,rank):
        super(Generator, self).__init__()
        self.rank=rank
        self.GC = ConvNet(self.rank,2,32,1)
        self.wave =WaveConvNeXt(self.rank,n_channels=8,G0=32,kSize=3)
        self.DC=Dataconsistency()
        self.WDC=Dataconsistency()
        self.dwt = DWT()
        self.iwt=IWT(self.rank)
        '''
        x:二次欠采样图像,是输入图像
        '''
    def forward(self,x,underimage,sub_mask,PD_label):
        inp=torch.cat([x,PD_label],dim=1)
        out=self.GC(inp)
        i_out = self.DC(out, underimage, sub_mask)  # under_image是永恒不变的，x是前一个模块的输出
        x_tar_wave2=self.dwt(i_out)
        x_refere_wave=self.dwt(PD_label)
        x_i=torch.cat([x_tar_wave2,x_refere_wave],dim=1)
        w_out=self.wave(x_i)
        w_out=w_out+x_tar_wave2
        w_out=self.iwt(w_out)
        x_dc = self.WDC(w_out,underimage, sub_mask)
        x_dc = torch.clamp(x_dc, 0, 1)
        return  x_dc