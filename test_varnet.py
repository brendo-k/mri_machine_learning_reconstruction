
# %%
from functools import partial
import os
import numpy as np

from ml_recon.models.varnet import VarNet
from ml_recon.models import Unet, ResNet, DnCNN, SwinUNETR
from torch.utils.data import DataLoader
import torch

from ml_recon.transforms import to_tensor, normalize
from ml_recon.dataset.sliceloader import SliceDataset 
from ml_recon.dataset.undersampled_decorator import UndersamplingDecorator
from ml_recon.utils import ifft_2d_img
from ml_recon.utils.root_sum_of_squares import root_sum_of_squares
from ml_recon.utils.evaluate import nmse, psnr
from ml_recon.Loss.ssim_loss import SSIMLoss
from train_varnet_self_supervised import to_device

from torchvision.transforms import Compose

def main():
    path = '/home/kadotab/python/ml/ml_recon/Model_Weights/self/20230525-173845VarNet.pt'
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/'
    backbone = partial(Unet, 2, 2)
    test(path, data_dir, backbone)

def test(weight_path, data_dir,  model):
    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transforms = Compose(
        (
            to_tensor(),
            normalize(),
        )
    )
 
    test_dataset = UndersamplingDecorator(
        SliceDataset(os.path.join(data_dir, 'multicoil_val')),
        transforms=transforms,
        R=4,
        R_hat=2
        )

    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1)

    checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])

    nmse_values = []
    ssim_values = []
    psnr_values = []
    ssim = SSIMLoss().to(device)

    model.eval()
    for data in test_loader:
        with torch.no_grad():
            mask, input, target, loss_mask = to_device(data, device, 'supervised')
            
            predicted_sampled = model(input, mask)
            predicted_sampled = predicted_sampled * (input == 0) + input
            predicted_sampled = ifft_2d_img(predicted_sampled)
            target = ifft_2d_img(target)

            target = root_sum_of_squares(target, coil_dim=1)
            predicted_sampled = root_sum_of_squares(predicted_sampled, coil_dim=1)

            nmse_values.append(nmse(predicted_sampled, target))
            ssim_values.append(ssim(predicted_sampled, target, target.max() - target.min()))
            psnr_values.append(psnr(predicted_sampled, target))
    
    ave_nmse = sum(nmse_values)/len(nmse_values)
    ave_ssim = 1 - sum(ssim_values)/len(ssim_values)
    ave_psnr = sum(psnr_values)/len(psnr_values)
    print(f'Average normalized mean squared error: {ave_nmse}')
    print(f'Average SSIM: {ave_ssim}')
    print(f'Average psnr: {ave_psnr}')
    return ave_nmse, ave_ssim, ave_psnr

if __name__ == "__main__":
    main()