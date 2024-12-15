import math
from typing import Tuple, List

import torch.nn as nn 
import torch
from ml_recon.models import Unet
import torch.nn.functional as F
from argparse import ArgumentParser

class PhaseAbsNetwork(nn.Module):
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
        self.abs_network = Unet(in_chan, out_chan, depth, chans, drop_prob)

    def forward(self, x):
        