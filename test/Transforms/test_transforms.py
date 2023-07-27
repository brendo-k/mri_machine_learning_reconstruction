from ml_recon.transforms import normalize, pad
from ml_recon.utils.fft_2d_img import ifft_2d_img

import torch



def test_normalize():
    # Generate random input data
    sample = {
        'undersampled': torch.rand((2, 3, 64, 64), dtype=torch.complex64),
        'k_space': torch.rand((2, 3, 64, 64), dtype=torch.complex64),
        'double_undersample': torch.rand((2, 3, 64, 64), dtype=torch.complex64)
    }

    normalizer = normalize()

    normalized_sample = normalizer(sample)

    k_max = normalized_sample['k_space'].abs().max()
    undersampled_max = normalized_sample['undersampled'].abs().max()
    double_under_max = normalized_sample['double_undersample'].abs().max()

    ft_under_max = ifft_2d_img(sample['undersampled']).abs().sum(1).sqrt().abs().max()

    assert k_max >= 1 
    assert undersampled_max >= 1
    assert double_under_max >= 1

    # Check if the shapes of input tensors are preserved
    assert normalized_sample['undersampled'].shape == sample['undersampled'].shape
    assert normalized_sample['k_space'].shape == sample['k_space'].shape

    assert torch.isclose(ft_under_max, normalized_sample['scaling_factor'])


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


   


