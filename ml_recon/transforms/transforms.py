import numpy as np
from ml_recon.utils import ifft_2d_img, fft_2d_img, root_sum_of_squares
import torch
import einops


def only_apply_to(sample, function, indecies):
    transofrmed_data = []
    for i, value in enumerate(sample):
        if i in indecies:
            transofrmed_data.append(function(value))
        else:
            transofrmed_data.append(value)

    return tuple(transofrmed_data)


class pad(object):
    def __init__(self, pad_dim):
        self.pad_dim = pad_dim

    def __call__(self, sample):
        return only_apply_to(sample, self.pad, indecies=[0, 1, 2])

    def pad(self, sample):
        x_diff = self.pad_dim[0] - sample.shape[-2]
        if x_diff < 0:
            x_diff = 0
        y_diff = self.pad_dim[1] - sample.shape[-1]
        if y_diff < 0:
            y_diff = 0
        pad = [(0, 0) for _ in range(sample.ndim)]
        pad[-2] = (x_diff//2, x_diff//2)
        pad[-1] = (y_diff//2, y_diff//2)

        sample = np.pad(sample, pad)
        return sample[..., :self.pad_dim[0], :self.pad_dim[1]]


class pad_recon(object):
    def __init__(self, pad_dim):
        self.pad_dim = pad_dim

    def __call__(self, sample):
        return only_apply_to(sample, self.pad, keys=['recon'])

    def pad(self, sample):
        x_diff = self.pad_dim[0] - sample.shape[-2]
        if x_diff < 0:
            x_diff = 0
        y_diff = self.pad_dim[1] - sample.shape[-1]
        if y_diff < 0:
            y_diff = 0
        pad = [(0, 0) for _ in range(sample.ndim)]
        pad[-2] = (x_diff//2, x_diff-x_diff//2)
        pad[-1] = (y_diff//2, y_diff-y_diff//2)

        sample = np.pad(sample, pad)
        return sample[..., :self.pad_dim[0], :self.pad_dim[1]]


class fft_2d(object):
    def __init__(self, axes):
        self.axes = axes

    def __call__(self, sample):
        return only_apply_to(sample, ifft_2d_img, keys=['undersampled', 'k_space'])


# Combines coils using naive rss with image intensity as highest point
class combine_coil(object):
    def __init__(self, coil_dim, use_abs=False):
        self.coil_dim = coil_dim
        self.use_abs = use_abs

    def __call__(self, sample: np.ndarray):
        sampled, undersampled = sample['k_space'], sample['undersampled']
        if self.use_abs:
            sampled = sampled.abs()
            undersampled = undersampled.abs()
        full_img = sampled.pow(2).sum(self.coil_dim).sqrt()
        undersampled_combined = undersampled.pow(2).sum(self.coil_dim).sqrt()

        sample['k_space'] = full_img
        sample['undersampled'] = undersampled_combined

        return sample

class to_tensor(object):
    def __call__(self, sample: np.ndarray):
        return only_apply_to(sample, torch.from_numpy, indecies=[0, 1, 2, 3])


# Normalize to [0, 1] range per contrast
class normalize(object):

    def __call__(self, sample):
        doub_under, under, sampled, k = sample

        image = root_sum_of_squares(ifft_2d_img(under.detach()), coil_dim=1)

        undersample_max = image.amax((1, 2), keepdim=True).unsqueeze(1)


        return (doub_under/undersample_max, under/undersample_max, sampled/undersample_max, k)


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
