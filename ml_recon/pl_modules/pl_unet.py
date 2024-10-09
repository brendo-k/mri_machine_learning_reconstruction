from ml_recon.models import Unet
from ml_recon.pl_modules.pl_ReconModel import plReconModel
from ml_recon.utils import ifft_2d_img, root_sum_of_squares

import torch


# define the LightningModule
class pl_Unet(plReconModel):
    def __init__(
            self, 
            contrast_order: list = [],
            chans: int = 18,
            lr: float = 1e-3
            ):

        super().__init__(contrast_order)

        self.save_hyperparameters()
        self.model = Unet(in_chan=4, out_chan=4, chans=chans)
        self.lr = lr
        self.contrast_order = contrast_order
        loss = torch.nn.MSELoss()
        self.loss = lambda target, prediction: loss(target, prediction)

    def training_step(self, batch, batch_idx):
        under, target = batch
        zf_img = root_sum_of_squares(ifft_2d_img(under, axes=[-1, -2]), coil_dim=2)
        target_img = root_sum_of_squares(ifft_2d_img(target, axes=[-1, -2]), coil_dim=2)

        estimate_target = self.model(zf_img)

        loss = self.loss(target_img, estimate_target)
        return loss


    def validation_step(self, batch, batch_idx):
        under, target = batch
        zf_img = root_sum_of_squares(ifft_2d_img(under, axes=[-1, -2]), coil_dim=2)
        target_img = root_sum_of_squares(ifft_2d_img(target, axes=[-1, -2]), coil_dim=2)

        estimate_target = self.model(zf_img)

        loss = self.loss(target_img, estimate_target)

        self.log('val_loss', loss, on_epoch=True)
        return loss


    def forward(self, data, mask): 
        return self.model(data)

    # optimizer configureation -> using adam w/ lr of 1e-3
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 6000, eta_min=1e-3) 
        return [optimizer]

