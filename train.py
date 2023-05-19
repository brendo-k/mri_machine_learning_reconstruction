from datetime import datetime

from torch.utils.data import DataLoader

from ml_recon.models.Unet import Unet
from ml_recon.transforms import (pad, combine_coil, toTensor, addChannels, 
                                   view_as_real, fft_2d, normalize, pad_recon, normalize_mean)
from ml_recon.utils.save_model import save_model
from ml_recon.utils import combine_coils, image_slices
from ml_recon.dataset.undersampled_slice_loader import UndersampledSliceDataset
from ml_recon.utils.collate_function import collate_fn

from torchvision.transforms import Compose
import numpy as np
import torch

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
        addChannels(),
        normalize_mean(), 
    )
)
dataset = UndersampledSliceDataset('/home/kadotab/header.json', transforms=transforms, R=4)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=6)
    

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# %%
model = Unet(1, 1, chans=32)
model.to(device)

# %%

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
    
                predicted_sampled = model(undersampled)

                loss = loss_function(predicted_sampled.unsqueeze(1), recon_slice)
                loss.backward()
                optimizer.step()
                    
                cur_loss += loss.item()*recon_slice.shape[0]
            writer.add_histogram('sens/weights1', model.down_sample_layers[0].conv1.weight, e)
            writer.add_histogram('castcade0/weights1', model.down_sample_layers[0].conv2.weight, e)
            writer.add_histogram('castcade0/weights2', model.down_sample_layers[3].conv.conv1.weight, e)
            writer.add_histogram('castcade0/weights11', model.up_sample_layers[3].conv.conv2.weight, e)
            writer.add_histogram('castcade0/weights12', model.conv1d.weight, e)
            writer.add_scalar('Loss/train', cur_loss/len(dataloader), e)
            print(f"Iteration: {e + 1:>d}, Loss: {cur_loss/len(dataloader):>7f}")
            cur_loss = 0

            if e % 10 == 9:
                save_model(model_weight_path, model, optimizer, e)
    except KeyboardInterrupt:
        pass

    save_model(model_weight_path, model, optimizer, -1)

# %%
train(model, loss_fn, optimizer, dataloader, epoch=50)


