import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Dict

def collate_fn(data: dict[str, torch.tensor]):
    # collate so coil dimension is same length so it can be 
    # passed through network
    k_space = collate_pad('k_space', data)
    double_undersample = collate_pad('double_undersample', data)
    undersampled = collate_pad('undersampled', data)

    # all the same size so no need to pad
    omega_mask = collate_stack('omega_mask', data)
    mask = collate_stack('mask', data)
    k = collate_stack('k', data)
    recon_rss = collate_stack('recon', data)
    scaling_factor = collate_stack('scaling_factor', data)
    undersampled_std = collate_stack('undersample_std', data)
    undersampled_mean = collate_stack('undersample_mean', data)
    double_undersampled_std = collate_stack('double_undersample_std', data)
    double_undersampled_mean = collate_stack('double_udnersample_mean', data)

    data = {
        'undersampled': undersampled, 
        'k_space': k_space,
        'mask': mask, 
        'recon': recon_rss,
        'double_undersample': double_undersample,
        'omega_mask': omega_mask, 
        'k': k,
        'scaling_factor': scaling_factor,
        'undersample_mean': undersampled_mean,
        'undersample_std': undersampled_std,
        'double_undersampled_mean': double_undersampled_mean,
        'double_undersampled_std': double_undersampled_std
    }
    return data

def collate_stack(key, data:Dict):
    if key not in data[0].keys():
        return None
    data = [d[key] for d in data]
    data_batched = torch.stack(data, dim=0)
    return data_batched

# takes the batch items and pads to largest size
def collate_pad(key, data):
    if key not in data[0].keys():
        return None
    data = [d[key] for d in data]
    data_padded = pad_sequence(data, batch_first=True)
    return data_padded