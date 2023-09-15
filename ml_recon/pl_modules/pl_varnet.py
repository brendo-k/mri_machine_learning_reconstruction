from torch import optim, nn
from ml_recon.models.varnet import VarNet
import pytorch_lightning as pl
import torch

from typing import Literal

# define the LightningModule
class pl_VarNet(pl.LightningModule):
    def __init__(
            self, 
                 in_chan: int, 
                 out_chan: int, 
                 loss_type: Literal['ssdu', 'supervised', 'k-weighted'],
                 num_cascades: int = 6, 
                 sens_chans: int = 8, 
                 model_chans: int = 18, 
                 dropout_prob: float = 0.0,
                 learning_rate: float = 1e-3
                 ):

        super().__init__()
        self.net = VarNet(
            in_chan,
            out_chan,
            num_cascades, 
            sens_chans, 
            model_chans,
            dropout_prob
        )
        self.lr = learning_rate
        self.loss_type = loss_type
        self.loss = self.complex_converter_decorator(torch.nn.functional.MSELoss)

    def training_step(self, batch, batch_idx):
        loss = self.choose_loss_type(self.loss_type, batch)
        return loss


    def validation_step(self, batch, batch_idx):
        sampled = batch['k_space']
        mask = batch['mask']
        undersampled = batch['undersampled']
        output = self.net(undersampled, mask)
        
        return {
            "batch_idx": batch_idx,
            "output": output,
            "val_loss": self.loss(
                output, sampled
            ),
        }

    # optimizer configureation -> using adam w/ lr of 1e-3
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

    # loss functions to use in training
    # ssdu: ssdu type training like in yaman paper
    # superivsed: supervised training like in varnet paper
    # k-weighted: k-weighted training like in charlies paper
    def choose_loss_type(self, type, data):
        assert type in ['ssdu', 'supervised', 'k-weighted']
        omega_mask = data['mask']
        undersampled = data['undersampled']
        if type == 'ssdu':
            lambda_mask = data['lambda_mask']
            doubly_undersampled = data['double_undersample']
            output = self.net(doubly_undersampled, lambda_mask * omega_mask)
            weighting = ~lambda_mask * omega_mask
            loss = self.loss(weighting * output, weighting * undersampled)
        elif type == 'supervised':
            output = self.net(undersampled, omega_mask)
            full_sampled = output['sampled']
            loss = self.loss(output, full_sampled)
        elif type == 'k-weighted':
            lambda_mask = data['lambda_mask']
            doubly_undersampled = data['double_undersample']
            output = self.net(doubly_undersampled, lambda_mask * omega_mask)
            weighting = ~lambda_mask * omega_mask / torch.sqrt(1 - data['K'])
            loss = self.loss(weighting * output, weighting * undersampled)

        return loss

    # this decorator takes in a loss function and returns a decorated loss function where 
    # input values can be complex. The complex values are simply conerted to real using
    # torch's view as real.
    def complex_converter_decorator(loss_function):
        def convert_to_complex(input1, input2):
            return loss_function(
                torch.view_as_real(input1),
                torch.view_as_real(input2)
            )
        return convert_to_complex