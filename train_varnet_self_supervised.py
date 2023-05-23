# %%
from datetime import datetime

from ml_recon.models.varnet import VarNet
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter

from ml_recon.transforms import (pad, toTensor, normalize)
from ml_recon.dataset.self_supervised_sampling import SelfSupervisedSampling
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
train_dataset = SelfSupervisedSampling(
    '/home/kadotab/train_16_header.json',
    transforms=transforms,
    R=4,
    R_hat=4
    )

val_dataset = SelfSupervisedSampling(
    '/home/kadotab/val_16_header.json',
    transforms=transforms,
    R=4,
    R_hat=4
    )

train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, num_workers=1)

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
model = VarNet(num_cascades=5)
model.to(device)
# %%
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
writer = SummaryWriter('/home/kadotab/scratch/runs/' + datetime.now().strftime("%Y%m%d-%H%M%S"))

# %%
path = '/home/kadotab/python/ml/ml_recon/Model_Weights/'


def train(model, loss_function, optimizer, dataloader):
    running_loss = 0
    for data in dataloader:
        sampled = data['undersampled']
        mask_lambda = data['lambda_mask']
        mask_omega = data['omega_mask']
        undersampled = data['double_undersampled']
        K = data['K']
        optimizer.zero_grad()
        mask_slice = mask_lambda.to(device)
        undersampled_slice = undersampled.to(device)
        sampled_slice = sampled.to(device)

        predicted_sampled = model(undersampled_slice, mask_slice)
        loss = loss_function(
            torch.view_as_real(predicted_sampled * (1 - mask_lambda) * (1 / torch.sqrt(1 - K))),
            torch.view_as_real(sampled_slice * (1 - mask_lambda) * (1 / torch.sqrt(1 - K)))
            )

        loss.backward()
        optimizer.step()
        running_loss += loss.item()*sampled.shape[0]
    return running_loss/len(dataloader)


def validate(model, loss_function, dataloader):
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

            val_running_loss += loss.item()*sampled.shape[0]
    return val_running_loss/len(dataloader)


for e in range(50):
    sample = next(iter(val_loader))
    output = model(sample['undersampled'].to(device), sample['mask'].to(device))
    output = ifft_2d_img(output)
    output = output.pow(2).sum(1).sqrt()
    output = output[:, 160:-160, :].cpu()
    diff = (output - sample['recon'])
    writer.add_image('val/recon', output[0].abs().unsqueeze(0)/output[0].abs().max(), e)
    writer.add_image('val/diff', diff[0].abs().unsqueeze(0)/diff[0].abs().max(), e)
    writer.add_image('val/target', sample['recon'][0].unsqueeze(0)/sample['recon'][0].max(), e)

    train_loss = train(model, loss_fn, optimizer, train_loader)
    val_loss = validate(model, loss_fn, val_loader)

    writer.add_scalar('train/loss', train_loss, e)
    writer.add_scalar('val/loss', val_loss, e)

    save_model(path, model, optimizer, e)
