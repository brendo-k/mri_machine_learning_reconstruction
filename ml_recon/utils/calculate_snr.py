from ml_recon.utils import k_to_img
import h5py
import numpy as np
import argparse

def main(args):
    file_name = args.file
    with h5py.File(file_name, 'r') as fr: 
        if 'k_space' in fr:
            k_space = fr['k_space'][()]
        elif 'kspace' in fr:
            k_space = fr['kspace'][()]
        else:
            raise ValueError(f'Could not find k-space keys: {fr.keys()}')
        if k_space.ndim == 4:
            k_space = k_space[None, ...]

        
    imgs = k_to_img(k_space, coil_dim=2)
    noise = imgs[..., :20, :20]
    noise_std = np.std(noise, axis=(-1, -2))
    
    circle = gen_circle(20, imgs.shape[-2:])

    print(imgs.shape)
    circle = np.tile(circle[None, None, :, :], (imgs.shape[0], imgs.shape[1], 1, 1))
    signal = imgs[circle]
    signal = np.reshape(signal, (imgs.shape[0], imgs.shape[1], -1)).mean(-1)

    snr = signal/noise_std

    for i in range(snr.shape[0]):
        print(snr[i])



def gen_circle(radius, grid_size) -> np.ndarray:
    y = np.arange(-grid_size[0]/2, grid_size[0]/2)
    x = np.arange(-grid_size[1]/2, grid_size[1]/2)

    xx, yy = np.meshgrid(x, y)

    circle = (xx**2 + yy**2) < radius**2
    print(circle.shape)

    return circle




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', '-f',  type=str)
    args = parser.parse_args()
    main(args)

