import matplotlib.pyplot as plt 
import nibabel as nib
import os
from ml_recon.transforms import normalize
from ml_recon.dataset.Brats_dataset import BratsDataset
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator
from ml_recon.utils import image_slices, ifft_2d_img, root_sum_of_squares

if __name__ == '__main__':
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/subset/'
    train_dataset = BratsDataset(os.path.join(data_dir, 'train'))
    val_dataset = BratsDataset(os.path.join(data_dir, 'val'))
    test_dataset = BratsDataset(os.path.join(data_dir, 'test'))

    undersampling_args = {
                'transforms': normalize()
            }
    
    train_dataset = UndersampleDecorator(train_dataset, **undersampling_args)
    val_dataset = UndersampleDecorator(val_dataset, **undersampling_args)
    test_dataset = UndersampleDecorator(test_dataset, **undersampling_args)


    #for i, value in enumerate(train_dataset):
    #    double, under, k_space, k = value
    #    images = ifft_2d_img(k_space)
    #    images = root_sum_of_squares(images, coil_dim=1)
    #    image_slices(images)
    #    plt.savefig('images/train/' + str(i))
    #    plt.close()

    for i, value in enumerate(val_dataset):
        double, under, k_space, k = value
        images = ifft_2d_img(k_space)
        images = root_sum_of_squares(images, coil_dim=1)
        image_slices(images)
        plt.savefig('images/val/' + str(i))
        plt.close()

    #for i, value in enumerate(test_dataset):
    #    double, under, k_space, k = value
    #    images = ifft_2d_img(k_space)
    #    images = root_sum_of_squares(images, coil_dim=1)
    #    image_slices(images)
    #    plt.savefig('images/test/' + str(i))
    #    plt.close()


