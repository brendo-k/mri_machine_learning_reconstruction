from monai.networks.nets import unetr
import torch.nn as nn
from argparse import ArgumentParser

class UnetR(nn.Module):
    def __init__(self, in_chan, out_chan, img_size, feature_size=8, hidden_size=256):
        super().__init__()
        self.network = unetr.UNETR(
                in_chan, 
                out_chan, 
                img_size, 
                feature_size=feature_size, 
                spatial_dims=2,
                hidden_size=hidden_size, 
                num_heads=6
                )

    def forward(self, x):
        x = self.network(x)
        return x
        
    @staticmethod
    def add_model_specific_args(parser: ArgumentParser): 
        parser.add_argument('--feature_size', type=int, default=16)
        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--hidden_size', type=int, default=256)
        parser.add_argument('--mlp_dim', type=int, default=256)
        return parser
