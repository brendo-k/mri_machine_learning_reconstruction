from ml_recon.dataset.kspace_brats import KSpaceBrats
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator
from ml_recon.utils import image_slices

if __name__ == '__main__':
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/chunked/train/'
    dataset = KSpaceBrats(data_dir, contrasts=['t1'])
    dataset = UndersampleDecorator(dataset)

    doub_under, under, k_space, k = dataset[0]

    print(under.shape)
    #image_slices(under.abs(), vmax=100)
    #plt.savefig('undersampled.png')
    #image_slices(doub_under.abs(), vmax=100)

    

