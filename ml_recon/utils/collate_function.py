import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Dict

def collate_fn(data):
    # collate so coil dimension is same length so it can be 
    # passed through network
    k_space = collate_pad(2, data)
    double_undersample = collate_pad(0, data)
    undersampled = collate_pad(1, data)
    K = collate_stack(3, data)

    return (double_undersample, undersampled, k_space, K)
    

def collate_stack(index, data:Dict):
    data = [d[index] for d in data]
    data_batched = torch.stack(data, dim=0)
    return data_batched

# takes the batch items and pads to largest size
def collate_pad(index, data):
    data = [d[index] for d in data]
    data_padded = pad_sequence(data, batch_first=True)
    return data_padded
