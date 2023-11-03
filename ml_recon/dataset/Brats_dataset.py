import os
from typing import Callable, Optional, Union, Collection
from argparse import ArgumentParser

from torch.utils.data import Dataset

from ml_recon.dataset.simulated_brats_dataset import SimulatedBrats
from ml_recon.dataset.kspace_brats import KSpaceBrats
from ml_recon.dataset.k_space_dataset import KSpaceDataset

class BratsDataset(KSpaceDataset):
    """
    Takes data directory and creates a dataset. Before using you need to specify the file reader 
    to use in the filereader variable. 
    """

    def __init__(
            self,
            data_dir: Union[str, os.PathLike], 
            nx:int = 256,
            ny:int = 256,
            contrasts: Collection[str] = ['t1', 't2', 'flair', 't1ce'], 
            transforms: Optional[Callable] = None,
            ):
        assert contrasts, 'Contrast list should not be empty!'

        super().__init__(nx=nx, ny=ny)

        sample_dir = os.listdir(data_dir)
        sample_dir.sort()
        sample_files = os.listdir(os.path.join(data_dir, sample_dir[0]))
        sample_files = [file for file in sample_files if 'label' not in file]
        file_path = os.path.join(data_dir, sample_dir[0], sample_files[0])
        _, extension = os.path.splitext(file_path)

        if extension == '.h5':
            self.data = KSpaceBrats(data_dir, nx=nx, ny=ny, contrasts=contrasts, transforms=transforms)
        elif extension == '.gz' or extension == '.npy':
            self.data = SimulatedBrats(data_dir, nx=nx, ny=ny, contrasts=contrasts, transforms=transforms)
        else:
            print(extension)
            raise ValueError(f"Can't load extension for {extension}")

        self.contrasts = self.data.contrasts
        self.contrast_order = self.data.contrast_order

        

    # length of dataset is the sum of the slices
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = KSpaceDataset.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False)

        parser.add_argument(
                '--data_dir', 
                type=str, 
                default='/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset_diff_phase/', 
                help=''
                )

        parser.add_argument(
                '--contrasts', 
                type=str, 
                nargs='+',
                default=['t1', 't2', 'flair', 't1ce'], 
                help=''
                )

        return parser

from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator
from ml_recon.utils import ifft_2d_img, root_sum_of_squares
if __name__ == '__main__':
    
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/subset/train'
    dataset = SimulatedBrats(data_dir)
    dataset = UndersampleDecorator(dataset)

    i = dataset[0]
    image = ifft_2d_img(i[2])
    image = root_sum_of_squares(image[0], coil_dim=0)
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap='gray')
    plt.savefig('image')
