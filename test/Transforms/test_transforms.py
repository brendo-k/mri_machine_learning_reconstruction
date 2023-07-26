from ml_recon.transforms import normalize, pad, normalize_mean
from ml_recon.utils import ifft_2d_img, fft_2d_img 
import torch


def test_normalize():
    x = torch.rand((10, 4, 640, 320), dtype=torch.complex64)
    sample = {'k_space': x}
    norm = normalize()
    x_norm = norm(sample)
    x_max = x_norm['k_space'].abs().max()
    x_min = x_norm['k_space'].abs().min()
    assert x_max == 1
    assert x_min >= 0
    assert x_min != 0

def test_pad():
    x = torch.rand(10, 4, 500, 300)
    sample = {'k_space': x}
    pad_func = pad((600, 400))
    x_pad = pad_func(sample)
    assert tuple(x_pad['k_space'].shape[-2:]) == (600, 400)
    assert (x_pad['k_space'][0, 0, :50, :50] == 0).all()
    assert (x_pad['k_space'][0, 0, -50:, -50:] == 0).all()
    assert (x_pad['k_space'][0, 0, 50:-50, 50:-50] != 0).all()
    assert (x_pad['k_space'][0, 0, 50:-50, 50:-50] != 0).all()
