# %%
from datetime import datetime

#from ml_recon.models.varnet import VarNet
from fastmri.models.varnet import VarNet
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

val_dataset = UndersampledSliceDataset(
    '/home/kadotab/val.json',
    transforms=transforms,
    R=4,
    )
    
train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, num_workers=1)

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
model = VarNet(num_cascades=3)
model.to(device)
# %%
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
writer = SummaryWriter('/home/kadotab/scratch/runs/' + datetime.now().strftime("%Y%m%d-%H%M%S"))

# %%
path = '/home/kadotab/python/ml/ml_recon/Model_Weights/'


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
        undersampled_slice = torch.view_as_real(undersampled_slice)

        predicted_sampled = model(undersampled_slice, (undersampled_slice != 0).bool().to(device))
        loss = loss_function(
            (predicted_sampled),
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
            undersampled_slice = torch.view_as_real(undersampled_slice)

            predicted_sampled = model(undersampled_slice, (undersampled_slice != 0).bool().to(device))
            loss = loss_function(
                (predicted_sampled),
                torch.view_as_real(sampled_slice)
                )

            val_running_loss += loss.item() * sampled.shape[0]
    return val_running_loss/len(dataloader)


for e in range(50):
    torch.use_deterministic_algorithms(True)
    sample = next(iter(val_loader))
    output = model(torch.view_as_real(sample['undersampled'].to(device)), (torch.view_as_real(sample['undersampled']) != 0).bool().to(device))
    output = torch.view_as_real(ifft_2d_img(torch.view_as_complex(output)))
    output = output.pow(2).sum(-1).sqrt().pow(2).sum(1).sqrt()
    output = output[:, 160:-160, :].cpu()
    diff = (output - sample['recon'])
    writer.add_image('val/recon', output[0].abs().unsqueeze(0)/output[0].abs().max(), e)
    writer.add_image('val/diff', diff[0].abs().unsqueeze(0)/diff[0].abs().max(), e)
    writer.add_image('val/target', sample['recon'][0].unsqueeze(0)/sample['recon'][0].max(), e)
    #writer.add_histogram('weights/lambda', model.lambda_reg)
    train_loss = train(model, loss_fn, optimizer, train_loader)
    val_loss = validate(model, loss_fn, val_loader)

    writer.add_scalar('train/loss', train_loss, e)
    writer.add_scalar('val/loss', val_loss, e)

    save_model(path, model, optimizer, e)
