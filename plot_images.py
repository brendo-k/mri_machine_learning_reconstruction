import os
import h5py
import matplotlib.pyplot as plt
from ml_recon.utils import ifft_2d_img, root_sum_of_squares

def plot(data, slice, type):
    data = root_sum_of_squares(ifft_2d_img(data), coil_dim=1)

    _, ax = plt.subplots(nrows=1, ncols=4)
    ax[0].imshow(data[0, :, :], cmap='gray')
    ax[1].imshow(data[1, :, :], cmap='gray')
    ax[2].imshow(data[2, :, :], cmap='gray')
    ax[3].imshow(data[3, :, :], cmap='gray')

    if not os.path.exists(os.path.join('images', type, file)):
        os.makedirs(os.path.join('images', type, file))

    plt.savefig(os.path.join('images', type, file, str(slice)))
    plt.close()

def plot_images(data, type):
    slices = [0, 10, 30, 50, -1]
    for slice in slices:
        data_slice = data[slice, :, :, :, :]
        plot(data_slice, slice, type)

if __name__ == '__main__':
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/with_labels/'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')


    for file in os.listdir(train_dir):
        file_names = os.path.join(train_dir, file, file + '.h5')

        with h5py.File(file_names, 'r') as fr:
            data = fr['k_space'][:]

        plot_images(data, 'train')


    for file in os.listdir(val_dir):
        file_names = os.path.join(val_dir, file, file + '.h5')

        with h5py.File(file_names, 'r') as fr:
            data = fr['k_space'][:]

        plot_images(data, 'val')

    for file in os.listdir(test_dir):
        file_names = os.path.join(test_dir, file, file + '.h5')

        with h5py.File(file_names, 'r') as fr:
            data = fr['k_space'][:]
        
        plot_images(data, 'test')



