
from ml_recon.Models.modl import modl
from torch.utils.data import DataLoader
from ml_recon.Transforms import (pad, trim_coils, combine_coil, toTensor, permute, 
                        view_as_real, remove_slice_dim, fft_2d, normalize, addChannels)
from ml_recon.Dataset.undersampled_dataset import UndersampledKSpaceDataset
from torchvision.transforms import Compose
import numpy as np

import torch
from ml_recon.Utils import image_slices
from ml_recon.Models.varnet import VarNet


 
torch.manual_seed(0)
np.random.seed(0)

 
transforms = Compose(
    (
        pad((640, 320)), 
        toTensor(),
        normalize(),
    )
)
dataset = UndersampledKSpaceDataset('/home/kadotab/projects/def-mchiew/kadotab/Datasets/fastMRI/multicoil_train', transforms=transforms, R=10)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    

 
data = next(iter(dataloader))

 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

 
model = VarNet(2, 2, num_cascades=12, use_norm=True)
model.to(device)

 
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.99, lr=0.0001)

 
from datetime import datetime

 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/tmp/kadota_runs/' +  datetime.now().strftime("%Y%m%d-%H%M%S"))

 
def train(model, loss_function, optimizer, dataloader, epoch=7):
    cur_loss = 0
    current_index = 0
    try:
        for e in range(epoch):
            for data in dataloader:

                sampled = data['sampled']
                mask = data['mask']
                undersampled = data['undersampled']
                for i in range(sampled.shape[0]):
                    optimizer.zero_grad()
                    sampled_slice = sampled[[i],...]
                    mask_slice = mask[[i],...]
                    undersampled_slice = undersampled[[i],...]
                    mask_slice = mask_slice.to(device)
                    undersampled_slice = undersampled_slice.to(device)
                    sampled_slice = sampled_slice.to(device)

                    predicted_sampled = model(undersampled_slice, mask_slice)
                    loss = loss_function(torch.view_as_real(predicted_sampled), torch.view_as_real(sampled_slice))

                    loss.backward()
                    optimizer.step()
                    cur_loss += loss.item()
                    current_index += 1
                    if current_index % 100 == 99:
                        writer.add_histogram('sens/weights1', next(model.sens_model.model.conv1d.parameters()), current_index)
                        writer.add_histogram('castcade0/weights1', next(model.cascade[0].unet.conv1d.parameters()), current_index)
                        writer.add_scalar('Loss/train', cur_loss, current_index)
                        print(f"Iteration: {current_index + 1:>d}, Loss: {cur_loss:>7f}")
                        cur_loss = 0
            model_name = model.__class__.__name__
            date = datetime.now().strftime("%Y%m%d-%H%M%S")
            torch.save(model.state_dict(), './Model_Weights/' + date + model_name + e +'.pt')
                 
    except KeyboardInterrupt:
        pass

    model_name = model.__class__.__name__
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), './Model_Weights/' + date + model_name + '.pt')

 
train(model, loss_fn, optimizer, dataloader)
