# %%
from ml_recon.Models.fastMRI_unet import Unet
from torch.utils.data import DataLoader
from ml_recon.Transforms import (pad, combine_coil, toTensor, addChannels, 
                                   view_as_real, fft_2d, normalize, pad_recon)
from ml_recon.Dataset.undersampled_slice_loader import UndersampledSliceDataset
from torchvision.transforms import Compose
import numpy as np
import torch
from ml_recon.Utils.collate_function import collate_fn
from datetime import datetime
from ml_recon.Utils.save_model import save_model
from ml_recon.Utils import combine_coils, image_slices
from ml_recon.Utils.load_checkpoint import load_checkpoint

# %%
torch.manual_seed(0)
np.random.seed(0)

# %%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/home/kadotab/scratch/runs/' +  datetime.now().strftime("%Y%m%d-%H%M%S"))

# %%

# %%
transforms = Compose(
    (
        pad((640, 320)),
        pad_recon((320, 320)), 
        fft_2d(axes=[-1, -2]),
        toTensor(),
        combine_coil(0, use_abs=True),
        normalize(), 
        addChannels(),
    )
)
dataset = UndersampledSliceDataset('/home/kadotab/header.json', transforms=transforms, R=4)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=6)

data = next(iter(dataloader))   

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
model = Unet(1, 1, chans=32)
model.to(device)

# %%
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=0.001,
        )

# %%
model_weight_path = '/home/kadotab/python/ml/ml_recon/Model_Weights/'
def train(model, loss_function, optimizer, dataloader, epoch=7):
    cur_loss = 0
    e = 0
    try:
        for e in range(epoch):
            for data in dataloader:
                undersampled = data['undersampled']
                recon = data['recon']
                optimizer.zero_grad()

                recon_slice = recon.to(device)
                undersampled = undersampled.to(device)
                undersampled = undersampled[:, 160:-160, :] 
                predicted_sampled = model(undersampled)

                loss = loss_function(predicted_sampled, recon_slice)
                    
                loss.backward()
                optimizer.step()
                    
                cur_loss += loss.item()
            writer.add_scalar('Loss/train', cur_loss, e)
            print(f"Iteration: {e + 1:>d}, Loss: {cur_loss:>7f}")
            save_model(model_weight_path, model, optimizer, e)
            cur_loss = 0
    except KeyboardInterrupt:
        pass

    save_model(model_weight_path, model, optimizer, -1)

# %%
train(model, loss_fn, optimizer, dataloader, epoch=50)


