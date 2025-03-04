import matplotlib.pyplot as plt
import numpy as np

def image_slices(data: np.ndarray, coil_num=0, vmin=None, vmax=None, cmap=None):
    """
    Helper function to plot multiple images. Dimensions assumed: Contrast, coil, height, width

    Args:
        data (np.ndarray): The input data array, expected to be 3D or 4D.
        coil_num (int, optional): The coil number to display. Defaults to 0.
        vmin (float, optional): The minimum data value that colormap covers. Defaults to None.
        vmax (float, optional): The maximum data value that colormap covers. Defaults to None.
        cmap (str or Colormap, optional): The colormap to use. Defaults to None.

    Raises:
        ValueError: If the input data does not have 3 or 4 dimensions.

    Returns:
        tuple: A tuple containing the figure and axes objects.
    """
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

    fig, axes = plt.subplots(width, height, figsize=(10, 10), squeeze=False)

    for ax in axes.flatten():
        ax.tick_params(
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                left=False,
                                labelleft=False,
                                labelbottom=False) 
    fig.subplots_adjust(wspace=0, hspace=0)

    for i in range(data.shape[0]):
        (x, y) = np.unravel_index(i, (width, height))
        current_slice = np.squeeze(data[i, coil_num, :, :])
        axes[x, y].imshow(current_slice, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')

    plt.tight_layout()

    return fig, axes

