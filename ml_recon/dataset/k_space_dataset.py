from argparse import ArgumentParser

from torch.utils.data import Dataset
import torch

class KSpaceDataset(Dataset):
    """
    Kspace dataset interface. Not sure how to make interfaces in python yet, but 
    here is a temporary one. It also can't enforce that the data is in k-space.
    Not sure how to do that either...
    """
    def __init__(self, nx, ny):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.contrast_order = []

    def __len__(self):
        pass

    def __getitem__(self, _) -> torch.Tensor:
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
                "--nx", 
                default=256,
                type=int,
                help="Number of points in the x direction"
                )

        parser.add_argument(
                "--ny", 
                default=256,
                type=int,
                help="Number of points in the y direction"
                )

        return parser
