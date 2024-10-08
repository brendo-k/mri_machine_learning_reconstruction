from ml_recon.dataset.self_supervised_decorator import SelfSupervisedDecorator
from ml_recon.utils import ifft_2d_img, root_sum_of_squares
from ml_recon.pl_modules.MRIDataModule import MRI_Loader


class UndersampledDataset(MRI_Loader):
    def __init__(
            self, 
            dataset_name: str,
            data_dir: str, 
            batch_size: int, 
            R: int = 4,
            R_hat: int = 2,
            contrasts: list[str] = ['t1', 't1ce', 't2', 'flair'],
            resolution: tuple[int, int] = (128, 128),
            line_constrained: bool = False,
            num_workers: int = 0,
            norm_method: str = 'k',
            segregated: bool = False
            ):
        super().__init__(dataset_name, data_dir, resolution, contrasts, batch_size=batch_size, num_workers=num_workers)
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resolution = resolution
        self.line_constrained = line_constrained
        self.segregated = segregated
        self.R = R
        self.R_hat = R_hat
        
        if norm_method == 'img':
            self.transforms = normalize_image_max()
        else: 
            self.transforms = normalize_k_max()


    def setup(self, stage):
        super().setup(stage)

        self.train_dataset = SelfSupervisedDecorator(
                self.train_dataset, 
                R=self.R, 
                R_hat=self.R_hat,
                line_constrained=self.line_constrained, 
                transforms=self.transforms, 
                segregated=self.segregated
                )

        self.val_dataset = SelfSupervisedDecorator(
                self.val_dataset, 
                R=self.R,
                R_hat=self.R_hat,
                line_constrained=self.line_constrained,
                transforms=self.transforms,
                segregated=self.segregated
                )

        self.test_dataset = SelfSupervisedDecorator(
                self.test_dataset, 
                R=self.R,
                R_hat=self.R_hat,
                line_constrained=self.line_constrained,
                transforms=self.transforms,
                segregated=self.segregated
                )

        self.contrast_order = self.train_dataset.contrast_order


class normalize_image_max(object):
    def __call__(self, data):
        undersampled, target = data
        img = root_sum_of_squares(ifft_2d_img(undersampled), coil_dim=1)
        scaling_factor = img.amax((1, 2), keepdim=True).unsqueeze(1)

        return (undersampled/scaling_factor, target/scaling_factor)

class normalize_k_max(object):
    def __call__(self, data):
        undersampled, target = data
        undersample_max = undersampled.abs().amax((1, 2, 3), keepdim=True)

        return (undersampled/undersample_max, target/undersample_max)
