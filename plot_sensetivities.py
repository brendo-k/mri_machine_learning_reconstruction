import matplotlib.pyplot as plt 
import nibabel as nib
import os
from ml_recon.transforms import normalize
from ml_recon.dataset.Brats_dataset import BratsDataset
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator
from ml_recon.models.SensetivityModel_mc import SensetivityModel_mc
from ml_recon.utils import image_slices, ifft_2d_img, root_sum_of_squares

if __name__ == '__main__':
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/subset/'
    val_dataset = BratsDataset(os.path.join(data_dir, 'val'))

    undersampling_args = {
                'transforms': normalize()
            }
    
    val_dataset = UndersampleDecorator(val_dataset, **undersampling_args)


    for i, value in enumerate(val_dataset):
        double, under, k_space, k = value
        masked_double = SensetivityModel_mc(2, 2, 8).mask(double, double != 0)
        image_double = ifft_2d_img(masked_double)
        for j in range(image_double.shape[0]):
            image_slices(image_double[j].abs())
            plt.savefig('images/sensetivity/' + str(i) + str(j))
            plt.close()
            image_slices(masked_double[j].abs(), vmax=0.01)
            plt.savefig('images/sensetivity/k-' + str(i) + str(j))
            plt.close()
        if i == 10:
            break
