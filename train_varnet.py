# %%
from ml_recon.Models.modl import modl
from torch.utils.data import DataLoader
from ml_recon.Transforms import (pad, toTensor, normalize)
from ml_recon.Dataset.undersampled_slice_loader import UndersampledSliceDataset
from torchvision.transforms import Compose
import numpy as np

import torch
from ml_recon.Utils import image_slices, save_model
from ml_recon.Utils.collate_function import collate_fn
from ml_recon.Models.varnet import VarNet

# %%
torch.manual_seed(0)
np.random.seed(0)

# %%
transforms = Compose(
    (
        pad((640, 320)), 
        toTensor(),
        normalize(),
    )
)
dataset = UndersampledSliceDataset('/home/kadotab/header.json', transforms=transforms, R=4)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    

# %%
data = next(iter(dataloader))

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
model = VarNet(2, 2, num_cascades=5, use_norm=True)
model.to(device)

# %%
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
writer = SummaryWriter('/home/kadotab/scratch/runs' +  datetime.now().strftime("%Y%m%d-%H%M%S"))

# %%
path = '/home/kadotab/python/ml/ml_recon/Model_Weights/'
def train(model, loss_function, optimizer, dataloader, epoch=7):
    cur_loss = 0
    try:
        for e in range(epoch):
            for data in dataloader:
                sampled = data['k_space']
                mask = data['mask']
                undersampled = data['undersampled']
                optimizer.zero_grad()
                mask_slice = mask.to(device)
                undersampled_slice = undersampled.to(device)
                sampled_slice = sampled.to(device)

                predicted_sampled = model(undersampled_slice, mask_slice)
                loss = loss_function(torch.view_as_real(predicted_sampled), torch.view_as_real(sampled_slice))

                loss.backward()
                optimizer.step()
                cur_loss += loss.item()
                  
            writer.add_histogram('sens/weights1', next(model.sens_model.model.conv1d.parameters()), e)
            writer.add_histogram('castcade0/weights1', next(model.cascade[0].unet.conv1d.parameters()), e)
            writer.add_histogram('castcade0/weights2', next(model.cascade[1].unet.conv1d.parameters()), e)
            writer.add_histogram('castcade0/weights11', next(model.cascade[-2].unet.conv1d.parameters()), e)
            writer.add_histogram('castcade0/weights12', next(model.cascade[-1].unet.conv1d.parameters()), e)
            writer.add_histogram('varnet/regularizer', model.lambda_reg.data, e)
            writer.add_scalar('Loss/train', cur_loss, e)
            print(f"Iteration: {e + 1:>d}, Loss: {cur_loss:>7f}")
            cur_loss = 0
            save_model(path, model, optimizer, e) 
    except KeyboardInterrupt:
        pass

    save_model(path, model, optimizer, -1)

# %%
train(model, loss_fn, optimizer, dataloader, 50)


