import torch

def load_checkpoint(path, model:torch.nn.Module, optimizer:torch.optim.Optimizer):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer
    