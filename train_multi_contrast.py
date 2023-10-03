from datetime import datetime
import os
import argparse
import time
import yaml
import json
import torch
from functools import partial
from ml_recon.utils import image_slices
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from ml_recon.models import Unet, ResNet, DnCNN, SwinUNETR
from ml_recon.models.varnet_mc import VarNet_mc
from ml_recon.dataset.Brats_dataset import BratsDataset 
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator
from ml_recon.utils import save_model, ifft_2d_img, root_sum_of_squares
from ml_recon.transforms import normalize
from torch.distributed import destroy_process_group

from torch.utils.data.distributed import DistributedSampler
from train_varnet_self_supervised import (
        to_device, 
        setup_devices,
        setup_scheduler,
        setup_ddp,
        train,
        validate, 
        to_device
        )


# Globals
PROFILE = False

def main():
    args = parser.parse_args()

    current_device, distributed = setup_devices(args.dist_backend, args.init_method, args.world_size)

    model = setup_model_backbone(args.model, current_device, chans=2*len(args.contrasts))

    model = setup_ddp(current_device, distributed, model)

    train_loader, val_loader = prepare_data(args, distributed)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer_dir = '/home/kadotab/scratch/runs/' + datetime.now().strftime("%m%d-%H:%M:%S") + model.__class__.__name__ + '-' + args.model + '-' + args.loss_type
    os.makedirs(os.path.join(writer_dir, 'weight_dir'))
    save_config(args, writer_dir)
    if current_device == 0: 
        writer = SummaryWriter(writer_dir)
    else: 
        writer = None

    scheduler = setup_scheduler(train_loader, optimizer, args.scheduler)
    for epoch in range(args.max_epochs):
        print(f'starting epoch: {epoch}')
        start = time.time()
        if distributed:
            train_loader.sampler.set_epoch(epoch) #pyright: ignore

        model.train()
        train_loss = train(model, loss_fn, train_loader, optimizer, current_device, args.loss_type, scheduler)
        model.eval()

        end = time.time()
        print(f'Epoch: {epoch}, loss: {train_loss}, time: {(end - start)/60} minutes')
        with torch.no_grad():
            plot_recon(model, train_loader, current_device, writer, epoch, args.loss_type, type='train')
            val_loss = validate(model, loss_fn, val_loader, current_device, args.loss_type)
            plot_recon(model, val_loader, current_device, writer, epoch, args.loss_type)
        
        
        if current_device == 0:
            if writer:
                writer.add_scalar('train/loss', train_loss, epoch)
                writer.add_scalar('val/loss', val_loss, epoch)
                if scheduler:
                    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

    save_model(os.path.join(writer_dir, 'weight_dir'), model, optimizer, args.max_epochs, current_device)

    if distributed:
        destroy_process_group()


def prepare_data(arg: argparse.Namespace, distributed: bool):
    data_dir = arg.data_dir
    
    train_dataset = BratsDataset(os.path.join(data_dir, 'train'), nx=arg.nx, ny=arg.ny, contrasts=arg.contrasts)

    val_dataset = BratsDataset(os.path.join(data_dir, 'val'), nx=arg.nx, ny=arg.ny, contrasts=arg.contrasts)

    undersampling_args = {
                'R': arg.R, 
                'R_hat': arg.R_hat, 
                'acs_lines': arg.acs_lines, 
                'poly_order': arg.poly_order,
                'transforms': normalize()
            }
    
    train_dataset = UndersampleDecorator(train_dataset, **undersampling_args)
    val_dataset = UndersampleDecorator(val_dataset, **undersampling_args)
    
    if arg.use_subset:
        train_dataset, _ = random_split(train_dataset, [0.1, 0.9])

    if distributed:
        print('Setting up distributed sampler')
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        shuffle = False
    else:
        val_sampler = None
        train_sampler = None 
        shuffle = True

    train_loader = DataLoader(train_dataset, 
                              batch_size=arg.batch_size,
                              num_workers=arg.num_workers,
                              shuffle=shuffle, 
                              sampler=train_sampler,
                              pin_memory=True,
                              )

    val_loader = DataLoader(val_dataset, 
                            batch_size=arg.batch_size, 
                            num_workers=arg.num_workers, 
                            sampler=val_sampler,
                            pin_memory=True,
                            )
    return train_loader, val_loader



def plot_recon(model, val_loader, device, writer, epoch, loss_type, type='val'):
    """ plots a single slice to tensorboard. Plots reconstruction, ground truth, 
    and error magnified by 4

    Args:
        model (nn.Module): model to reconstruct
        val_loader (nn.utils.DataLoader): dataloader to take slice
        device (str | int): device number/type
        writer (torch.utils.SummaryWriter): tensorboard summary writer
        epoch (int): epoch
    """
    if device == 0:
        # difference magnitude
        difference_scaling = 4

        # forward pass
        sample = next(iter(val_loader))
        mask, input_slice, target_slice, _, zf_mask = to_device(sample, device, 'supervised')
        
        output = model(input_slice, mask)
        output *= zf_mask
        output = output * (input_slice == 0) + input_slice
        output = output.cpu()
        
        sensetivity_maps = model.sens_model(input_slice, mask)
        writer.add_images(type + '-sense_map/image_0', sensetivity_maps[0, 0, :, :, :].cpu().abs().unsqueeze(1), epoch)
        writer.add_images(type + '-sense_map/image_1', sensetivity_maps[0, 1, :, :, :].cpu().abs().unsqueeze(1), epoch)
        writer.add_images(type + '-sense_map/image_2', sensetivity_maps[0, 2, :, :, :].cpu().abs().unsqueeze(1), epoch)
        writer.add_images(type + '-sense_map/image_3', sensetivity_maps[0, 3, :, :, :].cpu().abs().unsqueeze(1), epoch)
        # coil combination
        output = root_sum_of_squares(ifft_2d_img(output), coil_dim=2)
        ground_truth = root_sum_of_squares(ifft_2d_img(target_slice), coil_dim=2)
        x_input = root_sum_of_squares(ifft_2d_img(input_slice), coil_dim=2)
        assert isinstance(output, torch.Tensor)
        assert isinstance(ground_truth, torch.Tensor)
        assert isinstance(x_input, torch.Tensor)
        x_input = x_input.cpu()
        ground_truth = ground_truth.cpu()
        output = output.cpu()

        output = output[0]
        ground_truth = ground_truth[0]
        x_input = x_input[0]

        diff = (output - ground_truth).abs()

        # get scaling factor (skull is high intensity)
        image_scaling_factor = ground_truth.max()

        # scale images and difference
        image_scaled = output.abs()/image_scaling_factor
        diff_scaled = diff/(image_scaling_factor/difference_scaling)
        input_scaled = x_input.abs()/(image_scaling_factor)

        # clamp to 0-1 range
        image_scaled = image_scaled.clamp(0, 1)
        diff_scaled = diff_scaled.clamp(0, 1)
        input_scaled = input_scaled.clamp(0, 1)

        writer.add_images('images/' + type + '/recon', image_scaled.unsqueeze(1), epoch)
        writer.add_images('images/' + type + '/diff', diff_scaled.unsqueeze(1), epoch)
        writer.add_images('images/' + type + '/input', input_scaled.unsqueeze(1), epoch)

        if loss_type != 'supervised':
            mask_lambda_omega, _, _, loss_mask, zf_mask = to_device(sample, device, loss_type)
            writer.add_images('mask/' + type + '/omega_lambda_mask', mask_lambda_omega[0, :, [0], :, :], epoch)
            writer.add_images('mask/' + type + '/omega_mask', mask[0, :, [0], :, :], epoch)
            writer.add_images('mask/' + type + '/omega_not_lambda', loss_mask[0, :, [0], :, :].float(), epoch)
            
        # plot target if it's the first epcoh
        recon_scaled = ground_truth/image_scaling_factor
        recon_scaled = recon_scaled.clamp(0, 1)
        writer.add_images('images/' + type + '/target', recon_scaled.unsqueeze(1).abs(), epoch)


def setup_model_backbone(model_name, current_device, chans=8):
    if model_name == 'unet':
        backbone = partial(Unet, in_chan=chans, out_chan=chans, depth=4, chans=18)
    elif model_name == 'resnet':
        backbone = partial(ResNet, in_chan=chans, out_chan=chans, itterations=15, chans=32)
    elif model_name == 'dncnn':
        backbone = partial(DnCNN, in_chan=chans, out_chan=chans, feature_size=32, num_of_layers=15)
    elif model_name == 'transformer':
        backbone = partial(SwinUNETR, img_size=(128, 128), in_channels=2, out_channels=2, spatial_dims=2, feature_size=12)
        print('loaded swinunet!')
    else:
        raise ValueError(f'Backbone should be either unet resnet or dncnn but found {model_name}')

    model = VarNet_mc(backbone, num_cascades=6)
    params = sum([x.numel()  for x in model.parameters()])
    print(f'Model has {params:,}')
    model.to(current_device)

    return model


def save_config(args, writer_dir):
    args_dict = vars(args)
    with open(os.path.join(writer_dir, 'config.txt'), 'w') as f:
        f.write(json.dumps(args_dict, indent=4))


def load_config(args):
    """ loads yaml file """
    with open(args.config_file, 'r') as f:
        configs_dict = yaml.load(f)
        
    for k, v in configs_dict.items():
        setattr(args, k, v)

    return args

if __name__ == '__main__':

    # Argparse
    parser = argparse.ArgumentParser(description='Varnet self supervised trainer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate to use')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--max_epochs', type=int, default=50, help='')
    parser.add_argument('--num_workers', type=int, default=0, help='')
    parser.add_argument('--model', type=str, choices=['unet', 'resnet', 'dncnn', 'transformer'], default='unet')
    parser.add_argument('--loss_type', type=str, choices=['supervised', 'noiser2noise', 'ssdu', 'k-weighted'], default='ssdu')
    parser.add_argument('--scheduler', type=str, choices=['none', 'cyclic', 'cosine_anneal', 'steplr'], default='none')
    
    parser.add_argument('--init_method', default='tcp://localhost:18888', type=str, help='')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='')
    parser.add_argument('--world_size', default=2, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    parser.add_argument('--use_subset', action='store_true', help='')
    parser = BratsDataset.add_model_specific_args(parser)
    parser = UndersampleDecorator.add_model_specific_args(parser)
    main()
