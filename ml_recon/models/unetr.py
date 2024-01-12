from monai.networks.nets import unetr
import torch.nn as nn

class UnetR(nn.Module):
    def __init__(self, in_chan, out_chan, img_size, chans):
        super().__init__()
        self.network = unetr.UNETR(in_chan, out_chan, img_size,  feature_size=16, num_heads=4, hidden_size=256, mlp_dim=chans, spatial_dims=2)

    def forward(self, x):
        x = self.network(x)
        return x
        
