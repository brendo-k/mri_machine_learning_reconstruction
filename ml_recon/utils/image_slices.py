import matplotlib.pyplot as plt
import numpy as np

def image_slices(data: np.ndarray, coil_num=0, vmin=None, vmax=None, cmap=None):
    if data.ndim != 3 and data.ndim !=4:
        raise ValueError('Data needs 3 or 4 dimensions')

    if len(data.shape) == 3:
        data = data[:, np.newaxis, :, :]
    
    slices = data.shape[0]
    if slices == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        current_slice = np.squeeze(data[0, coil_num, :, :])
        axes.imshow(current_slice, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
        axes.tick_params(
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                left=False,
                                labelleft=False,
                                labelbottom=False) 
        return fig, axes

    width = int(np.ceil(np.sqrt(slices)))
    height = int(np.ceil(slices/width))

    fig, axes = plt.subplots(width, height, figsize=(10, 10))

    for i in range(data.shape[0]):
        (x, y) = np.unravel_index(i, (width, height))
        current_slice = np.squeeze(data[i, coil_num, :, :])
        axes[x, y].imshow(current_slice, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
        axes[x, y].tick_params(
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                left=False,
                                labelleft=False,
                                labelbottom=False) 
    plt.tight_layout()

    return fig, axes

