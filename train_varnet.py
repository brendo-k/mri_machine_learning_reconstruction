
# %%
from datetime import datetime

from ml_recon.models.varnet_resnet import VarNet
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.tensorboard import SummaryWriter

from ml_recon.transforms import pad, toTensor, normalize, pad_recon
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
        pad((640, 320)),
        toTensor(),
        normalize(),
        pad_recon((320, 320))
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

train_dataset, _ = random_split(train_dataset, [0.05, 0.95])
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=3, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=3)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = VarNet(num_cascades=5)
model.to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
tensorboard_dir = '/home/kadotab/scratch/runs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(tensorboard_dir)

# %%
path = '/home/kadotab/python/ml/ml_recon/Model_Weights/self2/'


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



sample = next(iter(val_loader))
for e in range(50):
    with torch.no_grad():
        sampled = sample['undersampled'].to(device)
        output = model(sampled, sample['mask'].to(device))
        output = ifft_2d_img(output)
        output = output.abs().pow(2).sum(1).sqrt()
        output = output[:, 160:-160, :].cpu()
        diff = (output - sample['recon'])
        writer.add_image('val/recon', output[0].unsqueeze(0)/output[0].max(), e)
        writer.add_image('val/diff', diff[0].abs().unsqueeze(0)/diff[0].abs().max(), e)
        writer.add_image('val/target', sample['recon'][0].unsqueeze(0)/sample['recon'][0].max(), e)
        writer.add_histogram('weights/lambda', model.lambda_reg)
        
    train_loss = train(model, loss_fn, optimizer, train_loader)
    val_loss = validate(model, loss_fn, val_loader)

    writer.add_scalar('train/loss', train_loss, e)
    writer.add_scalar('val/loss', val_loss, e)

save_model(path, model, optimizer, 50)
