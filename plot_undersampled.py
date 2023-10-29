from ml_recon.dataset.kspace_brats import KSpaceBrats
from ml_recon.dataset.simulated_brats_dataset import SimulatedBrats
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator
from ml_recon.utils import image_slices, ifft_2d_img
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/subset/train/'

    sim_brats = SimulatedBrats(data_dir)

    dataset = UndersampleDecorator(sim_brats)

    doub_under, under, k_space, k = dataset[0]

    plt.figure(1)
    plt.imshow(ifft_2d_img(k_space[0, 2]).abs(), cmap='gray')
    plt.figure(2)
    plt.imshow(ifft_2d_img(under[0, 2]).abs(), cmap='gray')
    plt.figure(3)
    plt.imshow(ifft_2d_img(doub_under[0, 2]).abs(), cmap='gray')
    plt.show()
    #image_slices(ifft_2d_img(doub_under[0]).abs())
    #plt.savefig('doub_undersampled.png')

    

    #doub_under, under, k_space, k = dataset[20]

    #image_slices(ifft_2d_img(under[0]).abs())
    #plt.savefig('undersampled2.png')
    #image_slices(ifft_2d_img(doub_under[0]).abs())
    #plt.savefig('doub_undersampled2.png')

    #doub_under, under, k_space, k = dataset[30]

    #image_slices(ifft_2d_img(under[0]).abs())
    #plt.savefig('undersampled3.png')
    #image_slices(ifft_2d_img(doub_under[0]).abs())
    #plt.savefig('doub_undersampled3.png')
