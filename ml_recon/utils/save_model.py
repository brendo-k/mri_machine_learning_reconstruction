from datetime import datetime
import torch


def save_model(path, model, optimizer, e, rank=0):
    if rank is None or rank == 0:
        torch.save({
            'model': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'epoch': e
            }, path + e + '.pt')