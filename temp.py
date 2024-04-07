from argparse import ArgumentParser
from ml_recon.pl_modules.pl_undersampled import UndersampledDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from ml_recon.pl_modules.pl_unet import pl_Unet
from ml_recon.utils import root_sum_of_squares, ifft_2d_img

import numpy as np
from ml_recon.dataset.undersample import get_mask_from_segregated_sampling, gen_pdf_bern, gen_pdf_columns, gen_pdf_columns_charlie
from functools import partial
import matplotlib.pyplot as plt


def main(args):
    data_module = UndersampledDataset(
            'brats', 
            args.data_dir, 
            batch_size=args.batch_size, 
            resolution=(args.ny, args.nx),
            num_workers=args.num_workers,
            norm_method=args.norm_method,
            R=args.R,
            line_constrained=args.line_constrained,
            self_supervsied=args.self_supervised
            ) 

    data_module.setup('train')
    test_dataset = data_module.test_dataset

    data = test_dataset[0]
    k_space = data['fs_k_space']

    imgs = root_sum_of_squares(ifft_2d_img(k_space), coil_dim=1)

    plt.imshow(imgs[0, :, :])
    plt.colorbar()
    plt.show()



if __name__ == '__main__': 
    parser = ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--learn_sampling', action='store_true')
    parser.add_argument('--line_constrained', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mask_method', type=str, default='all')
    parser.add_argument('--R', type=int, default=6)
    parser.add_argument('--lambda_param', type=float, default=0.)
    parser.add_argument('--limit_train_batches', type=float, default=1.0)
    parser.add_argument('--segregated', action='store_true')
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--ny', type=int, default=128)
    parser.add_argument('--norm_method', type=str, default='k')
    parser.add_argument('--self_supervised', action='store_true')
    parser.add_argument('--fd_param', type=float, default=0)
    parser.add_argument('--learn_R', action='store_true')
    parser.add_argument('--data_dir', type=str, default='/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset_random_phase/')
    
    args = parser.parse_args()

    main(args)
