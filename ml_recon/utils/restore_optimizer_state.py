from pytorch_lightning.callbacks import Callback
import torch
from torch.optim import Optimizer
class restore_optimizer(Callback):
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
    
    def on_train_start(self, trainer, pl_module):
        optimizer: Optimizer = trainer.optimizers[0] # type: ignore
        optimizer.load_state_dict(torch.load(self.checkpoint_path, weights_only=False)['optimizer_states'][0])
        old_lr = ''
        for param in optimizer.param_groups:
            old_lr = param['lr']
            param['lr'] = pl_module.lr
        print(f'Optimizer state restored from {self.checkpoint_path} with learning rate {pl_module.lr} from {old_lr}')
        