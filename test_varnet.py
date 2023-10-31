from functools import partial
import os
import numpy as np

from ml_recon.models.varnet_mc import VarNet_mc
from ml_recon.models import Unet
from torch.utils.data import DataLoader
import torch

from ml_recon.transforms import normalize
from ml_recon.dataset.Brats_dataset import BratsDataset 
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator 
from ml_recon.utils import ifft_2d_img
from ml_recon.utils.root_sum_of_squares import root_sum_of_squares
from ml_recon.utils.evaluate import nmse, psnr
from ml_recon.Loss.ssim_loss import SSIMLoss
from train_utils import to_device, setup_profile_context_manager
import matplotlib.pyplot as plt

from torchvision.transforms import Compose

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = '/home/kadotab/scratch/runs/6-t1,t2,flair-supervised/weight_dir/50.pt'
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/M4raw/multicoil_train_averaged/'

    model = setup_model(path, device)
    dataloader = setup_dataloader(data_dir)

    test(model, dataloader, num_contrasts=4, profile=False)

def setup_model(weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device)
    backbone = partial(Unet, 2, 2)
    model = VarNet_mc(backbone)
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

def test(model, test_loader, num_contrasts, profile):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    nmse_values = torch.zeros((num_contrasts, len(test_loader)))
    ssim_values = torch.zeros((num_contrasts, len(test_loader)))
    psnr_values = torch.zeros((num_contrasts, len(test_loader)))
    ssim = SSIMLoss().to(device)
    cm = setup_profile_context_manager(profile, 'test')

    model.eval()
    with cm as prof:
        for i, data in enumerate(test_loader):
            if prof:
                prof.step()
                if i >= (1 + 1 + 10) * 2:
                    break
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
                    nmse_values[contrast, i] = nmse(predicted_sampled[:, contrast, :, :].detach(), target[:, contrast, :, :].detach()).detach()
                    ssim_values[contrast, i] = ssim(predicted_sampled[:, [contrast], :, :].detach(), target[:, [contrast], :, :].detach(), target[:, contrast, :, :].max().detach() - target[:, contrast, :, :].min().detach()).detach()
                    psnr_values[contrast, i] = psnr(predicted_sampled[:, contrast, :, :].detach(), target[:, contrast, :, :].detach())
        
    ave_nmse = nmse_values.sum(1)/len(test_loader)
    ave_ssim = 1 - ssim_values.sum(1)/len(test_loader)
    ave_psnr = psnr_values.sum(1)/len(test_loader)
    print(f'Average normalized mean squared error: {ave_nmse}')
    print(f'Average SSIM: {ave_ssim}')
    print(f'Average psnr: {ave_psnr}')
    return ave_nmse, ave_ssim, ave_psnr

if __name__ == "__main__":
    main()
