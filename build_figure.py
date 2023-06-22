
from ml_recon.models.varnet_unet import VarNet
from torch.utils.data import DataLoader
import torch

from ml_recon.transforms import (toTensor, normalize)
from ml_recon.dataset.self_supervised_slice_loader import SelfSupervisedSampling
from ml_recon.utils import ifft_2d_img

from torchvision.transforms import Compose
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches   

path = '/home/kadotab/python/ml/ml_recon/Model_Weights/20230614-151243VarNet.pt'
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
test_dataset = SelfSupervisedSampling(
    '/home/kadotab/val_16_header.json',
    transforms=transforms,
    R=4,
    R_hat=4
    )

test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1)

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
model = VarNet(num_cascades=5)
model.to(device)

checkpoint = torch.load(path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])

model.train(False)
val_running_loss = 0
test_iter = iter(test_loader)
data = next(test_iter)
with torch.no_grad():
    sampled = data['k_space']
    mask = data['mask']
    undersampled = data['undersampled']
    mask_slice = mask.to(device)
    undersampled_slice = undersampled.to(device)
    sampled_slice = sampled.to(device)
    double_undersampled = data['double_undersample']

    predicted_sampled = model(undersampled_slice, mask_slice)
    predicted_sampled = predicted_sampled * data['scaling_factor'].to(device)
    predicted_sampled = ifft_2d_img(predicted_sampled)
    image = predicted_sampled.abs().pow(2).sum(1).sqrt()

plt.imshow(data['mask'][0, 160:-160], cmap='gray')
plt.savefig('mask.png')
plt.imshow(data['lambda_mask'][0, 160:-160], cmap='gray')
plt.savefig('set1.png')
plt.imshow(data['mask'][0, 160:-160] * ~data['lambda_mask'][0, 160:-160], cmap='gray')
plt.savefig('set2.png')

image = image[0, 160:-160, :]
image = image.flip(0)
image = image.cpu()
image_scaling = image.max()
plt.imshow(image, cmap='gray', vmax=image_scaling)
ax = plt.gca()
plt.colorbar()
plt.savefig('fully_supevised.png')

gt = data['recon'][0]
gt = gt.flip(0)
plt.imshow(gt, cmap='gray', vmax=image_scaling)
ax = plt.gca()
plt.savefig('gt.png')

plt.clf()
error = (gt - image).abs()
plt.imshow(error, cmap='gray')
plt.colorbar()
plt.savefig('error.png')
print(f'error sum: {(gt - image).pow(2).sum()}')
print(f' sum: {(gt).pow(2).sum()}')
print(error.pow(2).sum()/gt.pow(2).sum())