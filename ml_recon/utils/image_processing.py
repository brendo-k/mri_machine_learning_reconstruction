import numpy as np
import torch
from typing import Union

from numpy.typing import NDArray

def ifft_2d_img(data: torch.Tensor | NDArray, axes=[-1, -2]) -> Union[torch.Tensor, np.ndarray]:
    assert isinstance(data, torch.Tensor) or isinstance(data, np.ndarray), 'data should be a numpy array or pytorch tensor'

    if isinstance(data, torch.Tensor):
        data = torch.fft.ifftshift(data, dim=axes)
        data = torch.fft.ifft2(data, dim=axes, norm='ortho')
        data = torch.fft.fftshift(data, dim=axes)
    if isinstance(data, np.ndarray):
        data = np.fft.ifftshift(data, axes=axes)
        data = np.fft.ifft2(data, axes=axes, norm='ortho')
        data = np.fft.fftshift(data, axes=axes)
        data = data.astype(np.complex64)
        
    return data

def fft_2d_img(data: Union[torch.Tensor, NDArray], axes=[-1, -2]):
    if isinstance(data, torch.Tensor):
        data = torch.fft.ifftshift(data, dim=axes)
        data = torch.fft.fft2(data, dim=axes, norm='ortho')
        data = torch.fft.fftshift(data, dim=axes)
    if isinstance(data, np.ndarray):
        data = np.fft.ifftshift(data, axes=axes)
        data = np.fft.fft2(data, axes=axes, norm='ortho')
        data = np.fft.fftshift(data, axes=axes)
        data = data.astype(np.complex64)
    return data

def k_to_img(k_space, coil_dim=2, ft_dim=[-1, -2]):
    return root_sum_of_squares(ifft_2d_img(k_space, axes=ft_dim), coil_dim=coil_dim)

def root_sum_of_squares(data: Union[torch.Tensor, NDArray[np.complex64]], coil_dim=0):  # type: ignore
    """ Takes asquare root sum of squares of the abosolute value of complex data along the coil dimension

    Args:
        data (torch.Tensor): Data needed to be coil combined
        coil_dim (int, optional): dimension index. Defaults to 0.

    Returns:
        torch.Tensor: Coil combined data
    """
    assert data.ndim > coil_dim
    if isinstance(data, np.ndarray):
        return np.sqrt(np.sum(np.power(np.abs(data), 2), axis=coil_dim))
    elif isinstance(data, torch.Tensor):
        return torch.sqrt(data.abs().pow(2).sum(coil_dim) + 1e-20) # small value added to ensure sqrt is not zero
    else:
        raise ValueError(f'Data should be either a numpy array or pytorch Tensor')
    
