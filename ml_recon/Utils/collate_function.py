import torch

def collate_fn(data: dict[str, torch.tensor]):
    undersampled = [d['undersampled'] for d in data]
    k_space = [d['k_space'] for d in data]
    ismrmrd_header = [d['ismrmrd_header'] for d in data]
    mask = [d['mask'] for d in data]
    recon_rss = [d['reconstruction_rss'] for d in data]

    undersampled = torch.concat(undersampled, dim=0)
    k_space = torch.concat(k_space, dim=0)
    mask = torch.concat(mask, dim=0)

    data = {
        'undersampled': undersampled, 
        'k_space': k_space,
        'ismrmrd_header': ismrmrd_header,
        'mask': mask, 
        'recon': recon_rss,
    }
    return data