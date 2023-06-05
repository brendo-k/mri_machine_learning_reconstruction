import torch
from typing import Optional

def load_checkpoint(path, model:torch.nn.Module, optimizer : Optional[torch.optim.Optimizer] = None):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer
    