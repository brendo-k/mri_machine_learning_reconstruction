
# %%
from datetime import datetime
import os

from ml_recon.models.varnet_test import VarNet
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.tensorboard import SummaryWriter

from ml_recon.transforms import (pad, toTensor, normalize)
from ml_recon.dataset.undersampled_slice_loader import UndersampledSliceDataset
from ml_recon.utils import save_model, ifft_2d_img
from ml_recon.utils.collate_function import collate_fn

from torchvision.transforms import Compose
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
    '/home/kadotab/train.json',
    transforms=transforms,
    R=4,
    )



train_dataset, _ = random_split(train_dataset, [0.05, 0.95])
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = VarNet(num_cascades=5)
model.to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


profile_output = '/home/kadotab/scratch/runs/resnet5/'

def train(model, loss_function, optimizer, dataloader):
    model.train(True)
    running_loss = 0
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_output),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i, data in enumerate(dataloader):
            if i >= (1 + 1 + 3) * 2:
                break
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
            prof.step()

    return running_loss/len(dataloader)



        
train_loss = train(model, loss_fn, optimizer, train_loader)


