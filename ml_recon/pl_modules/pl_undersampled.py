from ml_recon.dataset.undersample_decorator import UndersampleDecorator
from ml_recon.utils import ifft_2d_img, root_sum_of_squares
from ml_recon.pl_modules.mri_module import MRI_Loader
from ml_recon.dataset.Brats_dataset import BratsDataset
from ml_recon.dataset.m4raw_dataset import M4Raw
from ml_recon.dataset.fastMRI_dataset import FastMRIDataset
import os


class UndersampledDataset(MRI_Loader):
    def __init__(
            self, 
            dataset_name: str,
            data_dir: str, 
            batch_size: int, 
            R: float = 4,
            R_hat: float = 2.0,
            contrasts: list[str] = ['t1', 't1ce', 't2', 'flair'],
            resolution: tuple[int, int] = (128, 128),
            line_constrained: bool = True,
            num_workers: int = 0,
            norm_method: str = 'k',
            segregated: bool = False,
            self_supervsied: bool = False
            ):

        super().__init__(dataset_name, data_dir, resolution, contrasts, batch_size=batch_size, num_workers=num_workers)
        self.save_hyperparameters()

        if dataset_name == 'brats': 
            self.dataset_class = BratsDataset
        elif dataset_name == 'fastmri':
            self.dataset_class = FastMRIDataset
        elif dataset_name == 'm4raw':
            self.dataset_class = M4Raw

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resolution = resolution
        self.line_constrained = line_constrained
        self.segregated = segregated
        self.R = R
        self.R_hat = R_hat
        self.self_supervised = self_supervsied
        
        if norm_method == 'img':
            self.transforms = normalize_image_max()
        else: 
            self.transforms = normalize_k_max()

    def setup(self, stage):
        super().setup(stage)
        data_dir = os.listdir(self.data_dir)

        train_file = next((name for name in data_dir if 'train' in name))
        val_file = next((name for name in data_dir if 'val' in name))
        test_file = next((name for name in data_dir if 'test' in name))

        train_dir = os.path.join(self.data_dir, train_file)
        val_dir = os.path.join(self.data_dir, val_file)
        test_dir = os.path.join(self.data_dir, test_file)

        self.train_dataset = self.dataset_class(
                train_dir, 
                nx=self.resolution[0], 
                ny=self.resolution[1],
                contrasts=self.contrasts,
                )

        self.val_dataset = self.dataset_class(
                val_dir, 
                nx=self.resolution[0], 
                ny=self.resolution[1],
                contrasts=self.contrasts,
                )

        self.test_dataset = self.dataset_class(
                test_dir, 
                nx=self.resolution[0], 
                ny=self.resolution[1], 
                contrasts=self.contrasts,
                )

        self.train_dataset = UndersampleDecorator(
                self.train_dataset,
                R=self.R,
                R_hat=self.R_hat,
                line_constrained=self.line_constrained,
                transforms=self.transforms,
                segregated=self.segregated,
                self_supervised=self.self_supervised
                )

        self.val_dataset = UndersampleDecorator(
                self.val_dataset,
                R=self.R,
                R_hat=self.R_hat,
                line_constrained=self.line_constrained,
                transforms=self.transforms,
                segregated=self.segregated,
                self_supervised=self.self_supervised
                )

        self.test_dataset = UndersampleDecorator(
                self.test_dataset,
                R=self.R,
                R_hat=self.R_hat,
                line_constrained=self.line_constrained,
                transforms=self.transforms,
                segregated=self.segregated,
                self_supervised=self.self_supervised
                )

        self.contrast_order = self.train_dataset.contrast_order


class normalize_image_max(object):
    def __call__(self, data):
        input = data['input']
        target = data['target']
        img = root_sum_of_squares(ifft_2d_img(input), coil_dim=1)
        scaling_factor = img.amax((1, 2), keepdim=True).unsqueeze(1)

        data.update({
            'input': input/scaling_factor, 
            'target': target/scaling_factor,
            'fs_k_space': data['fs_k_space']/scaling_factor
            })
        return data

class normalize_k_max(object):
    def __call__(self, data):
        input = data['input']
        target = data['target']
        undersample_max = input.abs().amax((1, 2, 3), keepdim=True)
        
        data.update({
            'input': input/undersample_max, 
            'target': target/undersample_max,
            'fs_k_space': data['fs_k_space']/undersample_max
            })
        return data
