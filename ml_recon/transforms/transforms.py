import numpy as np
from ml_recon.utils import ifft_2d_img, fft_2d_img, root_sum_of_squares
import torch
import einops


# Normalize to [0, 1] range per contrast
class normalize(object):
    def __init__(self, norm_method='mean'):
        self.norm_method=norm_method

    def __call__(self, sample):
        doub_under, under, sampled, k, omega_mask, lambda_mask = sample

        image = root_sum_of_squares(ifft_2d_img(under, axes=[-1, -2]), coil_dim=1)
        assert isinstance(image, torch.Tensor)

        if self.norm_method == 'mean':
            undersample_max = image.mean((1, 2), keepdim=True).unsqueeze(1)
        if self.norm_method == 'mean2':
            undersample_max = 2*image.mean((1, 2), keepdim=True).unsqueeze(1)
        if self.norm_method == 'k':
            undersample_max = under.abs().max((1, 2), keepdim=True).unsqueeze(1)
        elif self.norm_method == 'std':
            undersample_max = image.std((1, 2), keepdim=True).unsqueeze(1)
        elif self.norm_method == 'max':
            undersample_max = image.amax((1, 2), keepdim=True).unsqueeze(1)

        return (doub_under/undersample_max, under/undersample_max, sampled/undersample_max, k, omega_mask, lambda_mask)


# Normalize to [0, 1] range
class normalize_mean(object):

    def __call__(self, sample):
        doub_under, under, sampled, k = sample

        image = root_sum_of_squares(ifft_2d_img(under), coil_dim=1)
        undersample_max = image.mean()

        #undersample_min = image.min()
        
        #under = under_mask * fft_2d_img(ifft_2d_img(under) - undersample_min)
        #doub_under = doub_under_mask * fft_2d_img(ifft_2d_img(doub_under) - undersample_min)
        #sampled = fft_2d_img(ifft_2d_img(sampled) - undersample_min)

        under = under / undersample_max
        sampled = sampled / undersample_max
        doub_under = doub_under / undersample_max

        return (doub_under, under, sampled, k)
