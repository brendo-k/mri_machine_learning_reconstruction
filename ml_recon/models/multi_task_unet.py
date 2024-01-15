import torch.nn as nn 
import torch
from ml_recon.models.unet import Unet
from argparse import ArgumentParser

class MultiTaskUnet(nn.Module):
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
            initial_depth=4,
            initial_channel=8,
            joint_depth=4,
            joint_channel=8,
            ):

        super().__init__()
        self.initial_unet = torch.nn.ParameterList([Unet(2, 2, depth=initial_depth, chans=initial_channel) for _ in range(in_chan//2)])
        self.joint_unet = Unet(in_chan, out_chan, depth=joint_depth, chans=joint_channel)

    def forward(self, x):
        images = torch.split(x, 2, dim=1)
        inital_images = []
        for i, image in enumerate(images):
            initial_denoise = self.initial_unet[i](image)
            inital_images.append(initial_denoise)

        concated_images = torch.cat(inital_images, dim=1)
        final = self.joint_unet(concated_images)

        return final
            

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        parser.add_argument('--joint_chan', default=8, type=int)
        parser.add_argument('--joint_depth', default=4, type=int)
        parser.add_argument('--initial_depth', default=4, type=int)
        parser.add_argument('--initial_chan', default=8, type=int)
        return parser
