
# %%
from datetime import datetime

from ml_recon.models.varnet_resnet import VarNet
from torch.utils.data import DataLoader
import torch

from ml_recon.transforms import (toTensor, normalize)
from ml_recon.dataset.undersampled_slice_loader import UndersampledSliceDataset
from ml_recon.utils import ifft_2d_img
from ml_recon.utils.evaluate import nmse, psnr
from ml_recon.Loss.ssim_loss import SSIMLoss

from torchvision.transforms import Compose
import numpy as np

path = '/home/kadotab/python/ml/ml_recon/Model_Weights/self/20230525-173845VarNet.pt'
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
test_dataset = UndersampledSliceDataset(
    '/home/kadotab/test_16_header.json',
    transforms=transforms,
    R=4,
    )

test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1)

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
model = VarNet(num_cascades=5)
model.to(device)

checkpoint = torch.load(path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
nmse_values = []
ssim_values = []
psnr_values = []
ssim = SSIMLoss().to(device)
model.train(False)
val_running_loss = 0
for data in test_loader:
    with torch.no_grad():
        sampled = data['k_space']
        mask = data['mask']
        undersampled = data['undersampled']
        mask_slice = mask.to(device)
        undersampled_slice = undersampled.to(device)
        sampled_slice = sampled.to(device)

        predicted_sampled = model(undersampled_slice, mask_slice)
        predicted_sampled = ifft_2d_img(predicted_sampled)
        predicted_sampled *= data['scaling_factor'].to(device)
        sampled = ifft_2d_img(sampled)

        image = predicted_sampled.abs().pow(2).sum(1).sqrt()
        image = image[:, 160:-160, :] 
        gt = data['recon'] 
        gt = gt.to(device)
        nmse_values.append(nmse(gt, image))
        ssim_values.append(ssim(gt, image, gt.max()))
        psnr_values.append(psnr(gt, image))
    
print('Unet supevised')
print(sum(nmse_values)/len(nmse_values))
print(1 - sum(ssim_values)/len(ssim_values))
print(sum(psnr_values)/len(psnr_values))