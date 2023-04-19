
from ml_recon.Models.fastMRI_unet import Unet
from torch.utils.data import DataLoader
from ml_recon.Transforms import (pad, trim_coils, combine_coil, toTensor, permute, 
                                   view_as_real, fft_2d, normalize, pad_recon)
from ml_recon.Dataset.undersampled_dataset import UndersampledKSpaceDataset
from torchvision.transforms import Compose
import numpy as np
import torch
from ml_recon.Utils.collate_function import collate_fn
from datetime import datetime
from ml_recon.Utils.save_model import save_model


torch.manual_seed(0)
np.random.seed(0)


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/home/kadotab/scratch/runs/' +  datetime.now().strftime("%Y%m%d-%H%M%S"))




transforms = Compose(
    (
        pad((640, 320)),
        pad_recon((320, 320)), 
        fft_2d(axes=[2,3]),
        combine_coil(),
        toTensor(),
        normalize(), 
        view_as_real(),
        permute() 
    )
)
dataset = UndersampledKSpaceDataset('/home/kadotab/projects/def-mchiew/kadotab/Datasets/fastMRI/multicoil_train', transforms=transforms, R=4)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = Unet(2, 2)
model.to(device)


loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


model_weight_path = '/home/kadotab/python/ml/ml_recon/Model_Weights/'
def train(model, loss_function, optimizer, dataloader, epoch=7):
    cur_loss = 0
    current_index = 0
    try:
        for e in range(epoch):
            for data in dataloader:
                undersampled = data['undersampled']
                recon = data['recon']
                for i in range(undersampled.shape[0]):
                    optimizer.zero_grad()

                    recon_slice = torch.flip(recon[[i], ...].float(), (1, ))
                    undersampled_slice = undersampled[[i],...].float()

                    recon_slice = recon_slice.to(device)
                    undersampled_slice = undersampled_slice.to(device)
    
                    predicted_sampled = model(undersampled_slice)

                    predicted_sampled = torch.sqrt(predicted_sampled.sum(1).pow(2) + 1e-6)
                    assert not predicted_sampled.isnan().any()
                    predicted_sampled = predicted_sampled[:, 160:-160, :]
                    loss = loss_function(predicted_sampled, recon_slice)
                    
                    loss.backward()
                    optimizer.step()
                    
                    cur_loss += loss.item()
                    current_index += 1
                    if current_index % 1000 == 999:
                        writer.add_histogram('sens/weights1', next(model.down_sample_layers[0].conv1.parameters()), current_index)
                        writer.add_histogram('castcade0/weights1', next(model.down_sample_layers[0].conv2.parameters()), current_index)
                        writer.add_histogram('castcade0/weights2', next(model.down_sample_layers[3].conv.conv1.parameters()), current_index)
                        writer.add_histogram('castcade0/weights11', next(model.up_sample_layers[3].conv.conv2.parameters()), current_index)
                        writer.add_histogram('castcade0/weights12', next(model.conv1d.parameters()), current_index)
                        writer.add_scalar('Loss/train', cur_loss, current_index)
                        print(f"Iteration: {current_index + 1:>d}, Loss: {cur_loss:>7f}")
                        save_model(model_weight_path, model, optimizer, current_index)
                    cur_loss = 0
    except KeyboardInterrupt:
        pass

    save_model(model_weight_path, model, optimizer, -1)


train(model, loss_fn, optimizer, dataloader)


