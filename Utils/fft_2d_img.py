import numpy.fft as fft
import pyfftw


def fft_2d_img(data, axes=[2, 3]):
    data_shifted = fft.ifftshift(data, axes=axes)
    image = pyfftw.interfaces.numpy_fft.fft2(data_shifted, axes=axes)
    image = fft.fftshift(image, axes=axes)
    return image