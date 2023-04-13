import numpy as np
from ml_recon.Utils import combine_coils, fft_2d_img
import torch

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
        sample = sample[:, :self.coil_size, : ,:]
        return sample


class pad(object):
    def __init__(self, pad_dim):
        self.pad_dim = pad_dim

    def __call__(self, sample):
        return only_apply_to(sample, self.pad, keys=['undersampled', 'k_space', 'mask'])

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
        return only_apply_to(sample, fft_2d_img, keys=['undersampled', 'k_space'])


# Combines coils using naive rss with image intensity as highest point
class combine_coil(object):
    def __call__(self, sample: np.ndarray):
        sampled, undersampled = sample['k_space'], sample['undersampled']
        full_img = combine_coils(sampled, coil_dim=1)

        temp = sampled.transpose((1, 0, 2, 3))
        coil_sense = temp/full_img
        coil_sense = coil_sense.transpose((2, 0, 2, 3))
        undersampled_combined = np.sum(undersampled * coil_sense, axis=1)

        sample['k_space'] = full_img
        sample['undersampled'] = undersampled_combined

        return sample
        

class view_as_real(object):
    def __call__(self, sample: torch.Tensor):
        return only_apply_to(sample, torch.view_as_real, keys=['undersampled', 'k_space'])


class toTensor(object):
    def __call__(self, sample: np.ndarray):
        return only_apply_to(sample, torch.from_numpy, keys=['undersampled', 'k_space', 'mask', 'recon'])


class remove_slice_dim(object):
    def __call__(self, sample: np.ndarray):
        return only_apply_to(sample, self.remove_first_dim, keys=['undersampled', 'k_space'])

    def remove_first_dim(self, sample):
        return sample.reshape(((-1,) + sample.shape[2:]))
    

# Normalize to [0, 1] range
class normalize(object):
    def __call__(self, sample):
        return only_apply_to(sample, self.normalize, keys=['undersampled', 'k_space'])
    
    def normalize(self, sample):
        maximum = sample.abs().max()
        sample /= maximum
        return sample


class norm_normalize(object):
    def __call__(self, sample):
        return only_apply_to(sample, self.normalize, keys=['undersampled', 'k_space', 'recon'])
    
    def normalize(self, sample):
        sample_mean = sample.abs().mean((-1, -2), keepdim=True)
        sample_std = sample.abs().std((-1, -2), keepdim=True)
        return (sample - sample_mean)/sample_std


class permute(object):
    def __call__(self, sample):
        return only_apply_to(sample, self.permute, keys=['undersampled', 'k_space'])

    def permute(self, sample):
        return sample.permute(0, 3, 1, 2)

class addChannels(object):
    def __call__(self, sample):
        return only_apply_to(sample, self.add_dim)

    def add_dim(self, sample):
        return sample[:, None, :, :]
        
