import torch
import pytorch_lightning as pl 
from pytorch_lightning.loggers import CSVLogger
from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.dataset.Brats_dataset import BratsDataset
import os
from model import LOUPE


def main():
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/16_chans/multicoil_train/'
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset_random_phase/'
    test_dataset = BratsDataset(os.path.join(data_dir, 'test'), transforms=norm_loupe(), nx=128, ny=128)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=10)
    contrasts = test_dataset.contrast_order

    logger = CSVLogger("logs", name='same_prob_dist')
    model = LOUPE.load_from_checkpoint('lightning_logs/version_16575535/checkpoints/epoch=124-step=66750.ckpt', map_location='cpu', image_size=(4, 128, 128), R=8, contrasts=contrasts)
    trainer = pl.Trainer(logger=logger)
    trainer.test(model, test_loader)


class norm_loupe(object):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        image = root_sum_of_squares(ifft_2d_img(sample, axes=[-1, -2]), coil_dim=1)
        assert isinstance(image, torch.Tensor)

        undersample_max = image.amax((-1, -2), keepdim=True).unsqueeze(1)

        return sample/undersample_max

if __name__ == '__main__': 
    main()
