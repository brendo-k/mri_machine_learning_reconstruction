# %%
from ml_recon.models.varnet import VarNet
from torch.utils.data import DataLoader
import torch

from ml_recon.transforms import (pad, toTensor, normalize)
from ml_recon.dataset.undersampled_slice_loader import UndersampledSliceDataset
from ml_recon.utils import save_model
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
    )
)
dataset = UndersampledSliceDataset(
    '/home/kadotab/header.json',
    transforms=transforms,
    R=4,
    raw_sample_filter=lambda value: value['coils'] >= 16)

dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=1)
    
# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
model = VarNet(num_cascades=5)
model.to(device)
# %%
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

data = next(iter(dataloader))
# %%
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
writer = SummaryWriter('/home/kadotab/scratch/runs/' + datetime.now().strftime("%Y%m%d-%H%M%S"))

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
                cur_loss += loss.item()*sampled.shape[0]
                  
            writer.add_histogram('varnet/regularizer', model.lambda_reg.data, e)
            writer.add_scalar('train/loss', cur_loss/len(dataloader), e)
            print(f"Iteration: {e + 1:>d}, Loss: {cur_loss/len(dataloader):>7f}")
            cur_loss = 0
            if e % 10 == 9:
                save_model(path, model, optimizer, e) 
    except KeyboardInterrupt:
        pass

    save_model(path, model, optimizer, -1)


train(model, loss_fn, optimizer, dataloader, 50)


