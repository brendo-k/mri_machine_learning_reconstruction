from . import Unet
from torch import nn
from typing import Tuple 
import torch
from Utils import complex_conversion

class NormUnet(Unet):

    def __init__(
            self, 
            in_chan: int, 
            out_chan: int, 
            depth:int = 4,
            chans:int = 18, 
            with_instance_norm:bool = True,
            drop_prob:float = 0.0
    ):
        super().__init__(
            in_chan, 
            out_chan, 
            depth,
            chans, 
            with_instance_norm,
            drop_prob
        )

    # x should be (B, C, H, W)
    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape

        # seperate complex dimension
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def un_norm(self, x:torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        return x * std + mean

    def forward(self, x):
        x, mean, std = self.norm(x)

        x = super().forward(x)

        x = self.un_norm(x, mean, std)
        return x
        
