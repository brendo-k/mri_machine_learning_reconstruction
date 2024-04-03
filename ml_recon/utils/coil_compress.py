import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    coil_matrix = np.load('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/sens2.npy')
    coil_matrix = np.load('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/coil_compressed.npy')
    print(coil_matrix.shape)

    coil_matrix = coil_matrix.reshape(12, 256*256)

    u, s, vh = svd(coil_matrix, full_matrices=False)
   
    vh = vh[:8, :]
    vh = vh.reshape(-1, 256, 256)

    fig, ax = plt.subplots(4, 2)
    for i in range(vh.shape[0]):
        x, y = np.unravel_index(i, [4, 2])
        ax[x, y].imshow(np.abs(vh[i, :, :]))

    print(vh.shape)

    np.save('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/coil_compressed_8.npy', vh)

    vh = vh[:6, :]
    np.save('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/coil_compressed_6.npy', vh)

    vh = vh[:4, :]
    np.save('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/coil_compressed_4.npy', vh)
