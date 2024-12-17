import math
from typing import Tuple, List

import torch.nn as nn 
import torch
from ml_recon.models import Unet
import torch.nn.functional as F

class PhaseMagnitudeNetwork(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
            self, 
            in_chan, 
            out_chan, 
            depth=4,
            chans=18, 
            drop_prob=0
            ):

        super().__init__()
        self.phase_network = Unet(in_chan, out_chan, depth, chans, drop_prob)
        self.magnitude_network = Unet(in_chan, out_chan, depth, chans, drop_prob)

    def forward(self, x):
        b, c, h, w = x.shape
        magnitude_images = x[:, :c//2, ...]
        phase_images = x[:, c//2:, ...]
        output = torch.zeros_like(x)

        magnitude_output = self.magnitude_network(magnitude_images)
        phase_output = self.phase_network(phase_images)

        output[:, :c//2, :, :] = magnitude_output
        output[:, c//2:, :, :] = phase_output
        return output


        