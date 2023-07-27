import numpy as np
from ml_recon.utils import ifft_2d_img, fft_2d_img, root_sum_of_squares
import torch
import einops


def only_apply_to(sample, function, keys):
    for key in keys:
        if key not in sample.keys():
            continue
        sample[key] = function(sample[key])
    return sample


class trim_coils(object):
    def __init__(self, coil_size):
        self.coil_size = coil_size

    def __call__(self, sample):
        return only_apply_to(sample, self.trim_coil, keys=['undersampled', 'k_space'])

    def trim_coil(self, sample):
        sample = sample[:, :self.coil_size, :, :]
        return sample


class pad(object):
    def __init__(self, pad_dim):
        self.pad_dim = pad_dim

    def __call__(self, sample):
        return only_apply_to(sample, self.pad, keys=['undersampled', 'k_space', 'mask', 'omega_mask', 'double_undersample'])

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


class view_as_real(object):
    def __call__(self, sample: torch.Tensor):
        return only_apply_to(sample, self.complex_to_real, keys=['undersampled', 'k_space'])

    def complex_to_real(self, sample):
        sample = torch.view_as_real(sample)
        return einops.rearrange(sample, 'c h w cmplx-> (c cmplx) h w')


class toTensor(object):
    def __call__(self, sample: np.ndarray):
        return only_apply_to(sample, torch.from_numpy, keys=['undersampled', 'k_space', 'mask', 'recon', 'double_undersample', 'omega_mask', 'k', 'prob_omega'])


class remove_slice_dim(object):
    def __call__(self, sample: np.ndarray):
        return only_apply_to(sample, self.remove_first_dim, keys=['undersampled', 'k_space'])

    def remove_first_dim(self, sample):
        return sample.reshape(((-1,) + sample.shape[2:]))


# Normalize to [0, 1] range
class normalize(object):
    def __call__(self, sample):
        undersampled, k_space,  = sample['undersampled'], sample['k_space']
        undersample_max = root_sum_of_squares(ifft_2d_img(undersampled)).max()
        undersampled /= undersample_max
        k_space /= undersample_max
        if 'double_undersample' in sample.keys():
            sample['double_undersample'] = sample['double_undersample'] / undersample_max

        sample['undersampled'] = undersampled
        sample['k_space'] = k_space
        sample['scaling_factor'] = undersample_max
        return sample


# Normalize to [0, 1] range
class normalize_mean(object):
    def __init__(self, norm_chan):
        self.norm_chan = norm_chan

    def __call__(self, sample):
        undersampled, k_space = sample['undersampled'], sample['k_space']
        undersampled, undersampled_mean, undersampled_std = self.normalize(undersampled)
        k_space, _, _ = self.normalize(k_space)

        if 'double_undersample' in sample.keys():
            double_undersampled = sample['double_undersample']
            double_undersampled, double_undersampled_mean, double_undersampled_std = self.normalize(undersampled)
            sample['double_undersample'] = double_undersampled
            sample['double_undersample_mean'] = double_undersampled_mean
            sample['double_undersample_std'] = double_undersampled_std

        sample['undersampled'] = undersampled
        sample['k_space'] = k_space
        sample['undersample_mean'] = undersampled_mean
        sample['undersample_std'] = undersampled_std

        return sample
    
    def normalize(self, k_space):
        image = ifft_2d_img(k_space)
        image_real = image.real
        image_imag = image.imag

        image_mean_real = image_real.mean()
        image_mean_imag = image_imag.mean()

        image_real_std = image_real.std()
        image_imag_std = image_imag.std()

        image_real_norm = (image_real - image_mean_real)/image_real_std
        image_imag_norm = (image_imag - image_mean_imag)/image_imag_std

        k_space = fft_2d_img(image_real_norm + 1j * image_imag_nomr)
        
        return k_space, image_mean_real, image_std


class permute(object):
    def __call__(self, sample):
        return only_apply_to(sample, self.permute, keys=['undersampled', 'k_space'])

    def permute(self, sample):
        return sample.permute(0, 3, 1, 2)


class addChannels(object):
    def __call__(self, sample):
        return only_apply_to(sample, self.add_dim, keys=['undersampled', 'k_space'])

    def add_dim(self, sample):
        return sample.unsqueeze(0)


class convert_to_float(object):
    def __call__(self, sample):
        return only_apply_to(sample, self.convert_to_float, keys=['undersampled', 'k_space', 'recon'])
    
    def convert_to_float(self, sample):
        return sample.float()


