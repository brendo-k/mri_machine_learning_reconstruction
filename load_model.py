# %%
from ml_recon.Models.modl import modl
from torch.utils.data import DataLoader
from ml_recon.Transforms import (pad, trim_coils, combine_coil, toTensor, permute, 
                        view_as_real, remove_slice_dim, fft_2d, normalize, addChannels)
from ml_recon.Dataset.undersampled_dataset import UndersampledKSpaceDataset
from torchvision.transforms import Compose
import numpy as np

import torch
from ml_recon.Utils import image_slices, save_model
from ml_recon.Utils.collate_function import collate_fn
from ml_recon.Models.varnet import VarNet

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
dataset = UndersampledKSpaceDataset('/home/kadotab/projects/def-mchiew/kadotab/Datasets/fastMRI/multicoil_train', transforms=transforms, R=4)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
model = VarNet(2, 2, num_cascades=12, use_norm=True)
model.to(device)

# %%
checkpoint = torch.load('/home/kadotab/python/ml/ml_recon/Model_Weights/20230414-082106VarNet.pt', map_location=torch.device('cpu'))

# %%
model.load_state_dict(checkpoint['model'])

# %%
model.eval()

# %%
data = next(iter(dataloader))

# %%
output = model(data['undersampled'][[5], :, :, :], data['mask'][[5], :, :])


