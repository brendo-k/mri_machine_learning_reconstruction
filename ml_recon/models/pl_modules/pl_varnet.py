from torch import optim, nn
from torchvision.transforms import ToTensor
from ml_recon.models.varnet import VarNet
import pytorch_lightning as pl
from ml_recon.models.pl_modules.mri_module import MriModule
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch


# define the LightningModule
class pl_VarNet(MriModule):
    def __init__(self, 
                 in_chan, 
                 out_chan, 
                 num_cascades=6, 
                 sens_chans=8, 
                 model_chans=18, 
                 use_norm=True, 
                 dropout_prob=0):
        super().__init__()
        self.writer = SummaryWriter('/tmp/kadota_runs/' +  datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.net = VarNet(
            in_chan,
            out_chan,
            num_cascades, 
            sens_chans, 
            model_chans,
            use_norm, 
            dropout_prob
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        sampled = batch['k_space']
        mask = batch['mask']
        undersampled = batch['undersampled']
        loss = 0 
        for i in range(sampled.shape[0]):
            sampled_slice = sampled[[i],...]
            mask_slice = mask[[i],...]
            undersampled_slice = undersampled[[i],...]
            resampeld_estimate = self.net(undersampled_slice, mask_slice)
            loss += nn.functional.mse_loss(torch.view_as_real(resampeld_estimate), torch.view_as_real(sampled_slice))

        if(batch_idx % 99 == 100):
            self.writer.add_histogram('sens/weights1', next(self.net.sens_model.model.conv1d.parameters()), batch_idx)
            self.writer.add_histogram('castcade0/weights1', next(self.net.cascade[0].unet.conv1d.parameters()), batch_idx)
            self.writer.add_scalar('Loss/train', loss, batch_idx)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

