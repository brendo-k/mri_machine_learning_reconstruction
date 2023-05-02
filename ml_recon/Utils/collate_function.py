import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(data: dict[str, torch.tensor]):
    undersampled = [d['undersampled'] for d in data]
    k_space = [d['k_space'] for d in data]
    mask = [d['mask'] for d in data]
    recon_rss = [d['recon'] for d in data]

    undersampled = pad_sequence(undersampled, batch_first=True)
    k_space = pad_sequence(k_space, batch_first=True) 
    mask = torch.stack(mask, dim=0)
    recon_rss = torch.stack(recon_rss, dim=0) 

    data = {
        'undersampled': undersampled, 
        'k_space': k_space,
        'mask': mask, 
        'recon': recon_rss,
    }
    return data