from ml_recon.dataset.kspace_brats import KSpaceBrats
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator
from torch.utils.data import DataLoader
from ml_recon.transforms import normalize

if __name__ == '__main__':
    dataset = KSpaceBrats('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/with_labels/train', contrasts=['t1', 't1ce', 't2', 'flair'])

    undersampling_args = {
                'R': 4, 
                'R_hat': 2, 
                'acs_lines': 10, 
                'poly_order': 8,
                'transforms': normalize()
            }
    dataset = UndersampleDecorator(dataset, **undersampling_args)

    x = dataset[0]

