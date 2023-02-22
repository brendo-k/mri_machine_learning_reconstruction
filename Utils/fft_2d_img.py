import numpy as np
import torch
import pyfftw


def ifft_2d_img(data, axes=[2, 3]):
    if isinstance(data, torch.Tensor):
        ifft2 = torch.fft.ifft2
        ifftshift = torch.fft.ifftshift
        fftshift = torch.fft.fftshift
    if isinstance(data, np.ndarray):
        ifft2 = np.fft.ifft2
        ifftshift = np.fft.ifftshift
        fftshift = np.fft.fftshift
    data_shifted = ifftshift(data, axes=axes)
    image = ifft2(data_shifted, axes=axes)
    image = fftshift(image, axes=axes)
    return image

def fft_2d_img(data, axes=[2, 3]):
    if isinstance(data, torch.Tensor):
        fft2 = torch.fft.fft2
        ifftshift = torch.fft.ifftshift
        fftshift = torch.fft.fftshift
    if isinstance(data, np.ndarray):
        fft2 = np.fft.fft2
        ifftshift = np.fft.ifftshift
        fftshift = np.fft.fftshift
    data_shifted = fftshift(data, axes=axes)
    image = fft2(data_shifted, axes=axes)
    image = ifftshift(image, axes=axes)
    return image