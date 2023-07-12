from datetime import datetime
import torch


def save_model(path, model, optimizer, e, rank=0):
    if rank is None or rank == 0:
        model_name = model.__class__.__name__
        date = datetime.now().strftime("%Y%m%d-%H%M%S")
        torch.save({
            'model': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'epoch': e
            }, path + date + model_name + '.pt')