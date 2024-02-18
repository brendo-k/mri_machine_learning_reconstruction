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
from ml_recon.utils import ifft_2d_img, root_sum_of_squares, convert_weights_from_distributed
from ml_recon.utils.evaluate import nmse, psnr
from ml_recon.Loss.ssim_loss import SSIMLoss
from train_utils import to_device, setup_profile_context_manager
import matplotlib.pyplot as plt

from torchvision.transforms import Compose

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = '/home/kadotab/python/ismrm_figures/6-flair-ssdu/weight_dir/50.pt'
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset/'

    checkpoint = torch.load(path, map_location=device)

    new_dict = convert_weights_from_distributed(checkpoint)

    model = setup_model(new_dict)
    dataloader = setup_dataloader(data_dir, 'flair')

    test(model, dataloader, num_contrasts=1, profile=False)

def setup_model(weights):
    backbone = partial(Unet, 2, 2)
    model = VarNet_mc(backbone)
    model.load_state_dict(weights)
    return model

def setup_dataloader(data_dir, contrasts):
    transforms = Compose(
        (
            normalize(),
        )
    )
 
    test_dataset = UndersampleDecorator(
        BratsDataset(os.path.join(data_dir, 'test'), contrasts=contrasts),
        transforms=transforms,
        R=4,
        R_hat=2, 
        )
    
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1)
    return test_loader

def test(model, test_loader, profile, mask_output=True):
    np.random.seed(0)
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = test_loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    num_contrasts = dataset.contrasts

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
                predicted_sampled = root_sum_of_squares(predicted_sampled, coil_dim=2)

                target = ifft_2d_img(target)
                target = root_sum_of_squares(target, coil_dim=2)

                if mask_output:
                    output_mask = (target > 0.01)
                    target *= output_mask
                    predicted_sampled *= output_mask

                assert isinstance(predicted_sampled, torch.Tensor)
                assert isinstance(target, torch.Tensor)

                for contrast in range(input.shape[1]):
                    predicted_slice = predicted_sampled[:, [contrast], :, :]
                    target_slice = target[:, [contrast], :, :]

                    nmse_values[contrast, i] = nmse(target_slice, predicted_slice)
                    masked_ssim = ssim(target_slice, predicted_slice, target_slice.max(), reduced=False)
                    cur_ssim = masked_ssim[masked_ssim != 1].mean()
                    ssim_values[contrast, i] = cur_ssim
                    psnr_values[contrast, i] = psnr(target_slice, predicted_slice)
        
    ave_nmse = nmse_values.sum(1)/len(test_loader)
    ave_ssim = 1 - ssim_values.sum(1)/len(test_loader)
    ave_psnr = psnr_values.sum(1)/len(test_loader)
    print(f'Average normalized mean squared error: {ave_nmse}')
    print(f'Average SSIM: {ave_ssim}')
    print(f'Average psnr: {ave_psnr}')
    return ave_nmse, ave_ssim, ave_psnr

if __name__ == "__main__":
    main()
