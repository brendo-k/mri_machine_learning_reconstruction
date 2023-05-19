# %%
from ml_recon.Models.Unet import Unet
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
from ml_recon.Models.fastMRI_varnet import Varnet

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
        combine_coil(0),
        normalize(), 
        addChannels(),
        view_as_real(),
    )
)
dataset = UndersampledSliceDataset('/home/kadotab/header.json', transforms=transforms, R=2)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
model = Unet(2, 2)
model.to(device)

# %%
checkpoint = torch.load('/home/kadotab/python/ml/ml_recon/Model_Weights/20230505-204602Unet.pt')
model.load_state_dict(checkpoint['model'])

# %%
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
optimizer.load_state_dict(checkpoint['optimizer'])

# %%
data = next(iter(dataloader))

# %%
output = model(data['undersampled'].to(device))

# %%


# %%
output.shape

# %%
output.shape

# %%
output_abs = output.pow(2).sum(1).sqrt()

# %%
# %%
import matplotlib.pyplot as plt

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

                predicted_sampled = torch.sqrt(predicted_sampled.pow(2).sum(1)+ 1e-8)
                predicted_sampled = predicted_sampled[:, 160:-160, :]
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(predicted_sampled[0].cpu().detach())
                ax[1].imshow(recon_slice[0].cpu())
                ax[2].imshow(recon_slice[0].cpu() - predicted_sampled[0].detach().cpu())
                plt.show()
                print((recon_slice - predicted_sampled).pow(2).sum())
                loss = loss_function(predicted_sampled, recon_slice)
                print(loss) 
                loss.backward()
                optimizer.step()
                    
                cur_loss += loss.item()
            writer.add_histogram('sens/weights1', model.down_sample_layers[0].conv1.weight, e)
            writer.add_histogram('castcade0/weights1', model.down_sample_layers[0].conv2.weight, e)
            writer.add_histogram('castcade0/weights2', model.down_sample_layers[3].conv.conv1.weight, e)
            writer.add_histogram('castcade0/weights11', model.up_sample_layers[3].conv.conv2.weight, e)
            writer.add_histogram('castcade0/weights12', model.conv1d.weight, e)
            writer.add_scalar('Loss/train', cur_loss, e)
            print(f"Iteration: {e + 1:>d}, Loss: {cur_loss:>7f}")
            save_model(model_weight_path, model, optimizer, e)
            cur_loss = 0
    except KeyboardInterrupt:
        pass

    save_model(model_weight_path, model, optimizer, -1)

# %%
train(model, loss_fn, optimizer, dataloader, epoch=50)


