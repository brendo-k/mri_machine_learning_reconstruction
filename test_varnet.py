from functools import partial
import os
import numpy as np

from ml_recon.models.varnet import VarNet
from ml_recon.models import Unet
from torch.utils.data import DataLoader
import torch

from ml_recon.transforms import to_tensor, normalize
from ml_recon.dataset.Brats_dataset import BratsDataset 
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator 
from ml_recon.utils import ifft_2d_img
from ml_recon.utils.root_sum_of_squares import root_sum_of_squares
from ml_recon.utils.evaluate import nmse, psnr
from ml_recon.Loss.ssim_loss import SSIMLoss
from train_utils import to_device

from torchvision.transforms import Compose

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = '/home/kadotab/scratch/0910-08:57:44VarNet-unet-ssdu/80.pt'
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/'

    model = setup_model(path, device)
    dataloader = setup_dataloader(data_dir)

    test(model, dataloader, num_contrasts=4)

def setup_model(weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device)
    backbone = partial(Unet, 2, 2)
    model = VarNet(backbone)
    model.load_state_dict(checkpoint['model'])
    return model

def setup_dataloader(data_dir):
    transforms = Compose(
        (
            normalize(),
        )
    )
 
    test_dataset = UndersampleDecorator(
        BratsDataset(os.path.join(data_dir, 'test')),
        transforms=transforms,
        R=4,
        R_hat=2, 
        )
    
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1)
    return test_loader

def test(model, test_loader, num_contrasts):
    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    nmse_values = torch.zeros((num_contrasts, len(test_loader)))
    ssim_values = torch.zeros((num_contrasts, len(test_loader)))
    psnr_values = torch.zeros((num_contrasts, len(test_loader)))
    ssim = SSIMLoss().to(device)

    model.eval()
    for i, data in enumerate(test_loader):
        with torch.no_grad():
            mask, input, target, loss_mask, zero_filled = to_device(data, device, 'supervised')
            
            predicted_sampled = model(input, mask)
            predicted_sampled = predicted_sampled * (input == 0) + input
            
            predicted_sampled = ifft_2d_img(predicted_sampled)
            target = ifft_2d_img(target)

            target = root_sum_of_squares(target, coil_dim=2)
            predicted_sampled = root_sum_of_squares(predicted_sampled, coil_dim=2)

            assert isinstance(predicted_sampled, torch.Tensor)
            assert isinstance(target, torch.Tensor)

            for contrast in range(input.shape[1]):
                nmse_values[contrast, i] = nmse(predicted_sampled[:, contrast, :, :], target[:, contrast, :, :])
                ssim_values[contrast, i] = ssim(predicted_sampled[:, contrast, :, :], target[:, contrast, :, :], target[:, contrast, :, :].max() - target[:, contrast, :, :].min())
                psnr_values[contrast, i] = psnr(predicted_sampled[:, contrast, :, :], target[:, contrast, :, :])
    
    ave_nmse = nmse_values.sum(1)/len(test_loader)
    ave_ssim = 1 - ssim_values.sum(1)/len(test_loader)
    ave_psnr = psnr_values.sum(1)/len(test_loader)
    print(f'Average normalized mean squared error: {ave_nmse}')
    print(f'Average SSIM: {ave_ssim}')
    print(f'Average psnr: {ave_psnr}')
    return ave_nmse, ave_ssim, ave_psnr

if __name__ == "__main__":
    main()
