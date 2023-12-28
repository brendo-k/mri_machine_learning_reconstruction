from monai.networks.nets import unetr
import torch.nn as nn

class UnetR(nn.Module):
    def __init__(self, in_chan, out_chan, img_size, chans):
        self.network = unetr.UNETR(in_chan, out_chan, img_size,  feature_size=chans)

    def forward(self, x):
        return self.network(x)
