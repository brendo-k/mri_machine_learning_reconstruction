import numpy as np
import torch
from typing import Union

import numpy.typing as npt

def ifft_2d_img(data: Union[torch.Tensor, npt.NDArray[np.complex_]], axes=[-1, -2]) -> Union[torch.Tensor, np.ndarray]:
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

def fft_2d_img(data: Union[torch.Tensor, np.ndarray], axes=[-1, -2]):
    assert isinstance(data, torch.Tensor) or isinstance(data, np.ndarray), 'data should be a numpy array or pytorch tensor'
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
