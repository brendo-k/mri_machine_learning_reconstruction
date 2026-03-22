
from net.ConvNextMR import *
from wavelet_transform import DWT as dwt

class ISTANetPlus(nn.Module):
    def __init__(self,rank,num_layers):
        super(ISTANetPlus, self).__init__()
        self.rank=rank
        self.num_layers = num_layers
        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(Generator(self.rank))
        self.layers = nn.ModuleList(self.layers)
    def forward(self,inp,sub_mask,PD_label):
        x = inp
        for i in range(self.num_layers):
            x= self.layers[i](x,inp,sub_mask,PD_label)
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
