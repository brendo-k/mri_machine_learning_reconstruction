import numpy as np
import torch
import pyfftw


def ifft_2d_img(data, axes=[-1, -2]):
    assert isinstance(data, torch.Tensor) or isinstance(data, np.ndarray), 'data should be a numpy array or pytorch tensor'

    if isinstance(data, torch.Tensor):
        data_shifted = torch.fft.ifftshift(data, dim=axes)
        image = torch.fft.ifft2(data_shifted, dim=axes, norm='ortho')
        image = torch.fft.fftshift(image, dim=axes)
    if isinstance(data, np.ndarray):
        data_shifted = np.fft.ifftshift(data, axes=axes)
        image = np.fft.ifft2(data_shifted, axes=axes)
        image = np.fft.fftshift(image, axes=axes)
        
    return image

def fft_2d_img(data, axes=[2, 3]):
    if isinstance(data, torch.Tensor):
        data_shifted = torch.fft.fftshift(data, dim=axes)
        image = torch.fft.fft2(data_shifted, dim=axes, norm='ortho')
        image = torch.fft.ifftshift(image, dim=axes)
    if isinstance(data, np.ndarray):
        data_shifted = np.fft.fftshift(data, axes=axes)
        image = np.fft.fft2(data_shifted, axes=axes)
        image = np.fft.ifftshift(image, axes=axes)
    return image