
# %%
from datetime import datetime

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

from ml_recon.models.varnet import VarNet
from ml_recon.transforms import (pad, toTensor, normalize)
from ml_recon.dataset.undersampled_slice_loader import UndersampledSliceDataset
from ml_recon.utils.collate_function import collate_fn

import numpy as np


# %%
torch.manual_seed(0)
np.random.seed(0)

# %%
transforms = Compose(
    (
        # pad((640, 320)),
        toTensor(),
        normalize(),
    )
)
train_dataset = UndersampledSliceDataset(
    '/home/kadotab/train_16_header.json',
    transforms=transforms,
    R=4,
    )

val_dataset = UndersampledSliceDataset(
    '/home/kadotab/val_16_header.json',
    transforms=transforms,
    R=4,
    )

train_dataset, _ = random_split(train_dataset, [0.001, 0.999])
train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, num_workers=1)

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = next(iter(val_loader))
# %%
model = VarNet(num_cascades=5)
model.to(device)
# %%
loss_fn = torch.nn.MSELoss()


# %%

def train(model, loss_function, optimizer, dataloader):
    model.train(True)
    running_loss = 0
    for data in dataloader:
        sampled = data['k_space']
        mask = data['mask']
        undersampled = data['undersampled']
        optimizer.zero_grad()
        mask_slice = mask.to(device)
        undersampled_slice = undersampled.to(device)
        sampled_slice = sampled.to(device)

        predicted_sampled = model(undersampled_slice, mask_slice)
        loss = loss_function(
            torch.view_as_real(predicted_sampled),
            torch.view_as_real(sampled_slice)
            )

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * sampled.shape[0]
    return running_loss/len(dataloader)


def validate(model, loss_function, dataloader):
    model.train(False)
    val_running_loss = 0
    for data in dataloader:
        with torch.no_grad():
            sampled = data['k_space']
            mask = data['mask']
            undersampled = data['undersampled']
            mask_slice = mask.to(device)
            undersampled_slice = undersampled.to(device)
            sampled_slice = sampled.to(device)

            predicted_sampled = model(undersampled_slice, mask_slice)
            loss = loss_function(
                torch.view_as_real(predicted_sampled),
                torch.view_as_real(sampled_slice)
                )

            val_running_loss += loss.item() * sampled.shape[0]
    return val_running_loss/len(dataloader)


lr = torch.linspace(1e-8, 0.1, 400)
losses = []
sample = next(iter(train_loader))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train(model, loss_fn, optimizer, train_loader)
for lrs in lr:
    optimizer = torch.optim.Adam(model.parameters(), lr=lrs)
    output = model(sample['undersampled'].to(device), sample['mask'].to(device))
    loss = loss_fn(torch.view_as_real(output), torch.view_as_real(sample['k_space']).to(device))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

import numpy as np
plt.plot(lr, losses)
plt.yscale('log')
plt.xscale('log')
plt.savefig('loss_vs_lr.png')

    

