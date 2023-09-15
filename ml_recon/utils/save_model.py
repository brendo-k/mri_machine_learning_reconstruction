import torch
from typing import Union

def save_model(path: str, model: torch.nn.Module, optimizer:torch.optim.Optimizer, e:int, rank:Union[int, str]=0):
    if rank is None or rank == 0:
        torch.save({
            'model': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'epoch': e
            }, path + str(e) + '.pt')
