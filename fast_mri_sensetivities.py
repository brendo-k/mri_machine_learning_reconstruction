import matplotlib.pyplot as plt 
import numpy as np
import nibabel as nib
import os
from ml_recon.transforms import normalize
from ml_recon.dataset.Brats_dataset import BratsDataset
from ml_recon.dataset.fastMRI_dataset import FastMRIDataset
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator
from ml_recon.models.SensetivityModel_mc import SensetivityModel_mc
from ml_recon.utils import image_slices, ifft_2d_img, root_sum_of_squares, fft_2d_img
from ml_recon.utils.espirit import espirit
import h5py 

if __name__ == '__main__':
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/16_chans/'
    val_dataset = FastMRIDataset(os.path.join(data_dir, 'multicoil_val'))

    undersampling_args = {
                'transforms': normalize()
            }
    
    val_dataset = UndersampleDecorator(val_dataset, **undersampling_args)

    value = val_dataset[2]
    double, under, k_space, k = value

    plt.imshow(root_sum_of_squares(ifft_2d_img(k_space), coil_dim=1)[0])
    plt.savefig('img')
    plt.close()
    maps = espirit(k_space.permute(0, 2, 3, 1).numpy(), 6, 24, 0.2, 0.90) 
    image_slices(np.abs(maps)[0, :, :, :, 0].transpose((2, 0, 1)))
    plt.savefig('maps')
    plt.close()

    maps = maps[0, :, :, :, 0].transpose((2, 0, 1))
    
    recon = (k_space * np.conj(maps)).sum(1) / (np.sum(np.conj(maps)*maps, 0) + 1e-10)
    plt.imshow(np.angle(recon[0]))
    plt.savefig('phase')
    plt.imshow(np.abs(fft_2d_img(np.angle(recon[0]))))
    plt.savefig('phase_fft')
    plt.imshow(np.abs(recon[0]))
    plt.savefig('image')
