import h5py
import numpy as np
import matplotlib.pyplot as plt
from ml_recon.utils import image_slices, root_sum_of_squares, ifft_2d_img
import argparse



def main():
    parser = argparse.ArgumentParser(description='My command-line script')
    parser.add_argument('-f', '--filename', type=str, help='filename', required=True)
    parser.add_argument('--images', action='store_true', help='flag for image_space')
    parser.add_argument('-c', '--contrast_index', type=int, help='index for contrast to plot', default=0)
    parser.add_argument('-s', '--slice_index', type=int, help='slice index to plot')

    args = parser.parse_args()

    with h5py.File(args.filename) as fr:
        if 'k_space' in fr:
            kspace = fr['k_space'][:]
        if 'kspace' in fr:
            kspace = fr['kspace'][:]
    kspace.astype(np.complex64)
    if args.slice_index:
        kspace = kspace[[args.slice_index]]

    print(kspace.shape)
    if args.images: 
        plotting_aray = root_sum_of_squares(ifft_2d_img(kspace).astype(np.complex64), coil_dim=2)[:, args.contrast_index]
    else: 
        plotting_aray = np.abs(kspace[:, args.contrast_index, 0, :, :])**0.02

    image_slices(plotting_aray, cmap='gray')
    plt.show()

        


if __name__ == "__main__":
    main()


