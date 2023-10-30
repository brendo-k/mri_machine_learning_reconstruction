from ml_recon.dataset.kspace_brats import KSpaceBrats
from ml_recon.dataset.simulated_brats_dataset import SimulatedBrats
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator
from ml_recon.utils import image_slices, ifft_2d_img, root_sum_of_squares
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset/val/'

    sim_brats = KSpaceBrats(data_dir)

    dataset = UndersampleDecorator(sim_brats)

    doub_under, under, k_space, k = dataset[10]

    plt.figure(1)
    image_slices(root_sum_of_squares(ifft_2d_img(k_space), coil_dim=1).abs(), cmap='gray')
    plt.figure(2)
    image_slices(ifft_2d_img(under[0]).abs(), cmap='gray')
    plt.figure(3)
    image_slices(ifft_2d_img(doub_under[0]).abs(), cmap='gray')
    plt.figure(4)
    


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
