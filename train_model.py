from datetime import datetime
import re 
import os
import argparse
import time
import yaml
import torch
from functools import partial

import torch
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from ml_recon.models import Unet, ResNet, DnCNN, SwinUNETR, UnetR, MultiTaskUnet
from ml_recon.models.varnet_mc import VarNet_mc
from ml_recon.dataset.m4raw_dataset import M4Raw 
from ml_recon.dataset.kspace_brats import KSpaceBrats
from ml_recon.dataset.Brats_dataset import BratsDataset
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator
from ml_recon.utils import save_model, ifft_2d_img, root_sum_of_squares
from ml_recon.transforms import normalize
from ml_recon.losses import L1L2Loss
from test_varnet import test

from train_utils import (
        to_device, 
        save_config,
        setup_devices,
        setup_scheduler,
        setup_ddp,
        train,
        validate, 
        )


# Globals
PROFILE = False

def main(args):
    print("Starting code")

    current_device, distributed = setup_devices(args.dist_backend, args.init_method, args.world_size)

    model = setup_model_backbone(args)
    model.to(current_device)

    model = setup_ddp(current_device, distributed, model)

    train_loader, val_loader, test_loader = prepare_data(args, distributed)

    #loss_fn = torch.nn.MSELoss()
    loss_fn = L1L2Loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = setup_scheduler(train_loader, optimizer, args.scheduler)

    writer, writer_dir, run_identifier = create_writer(current_device, args)

    for epoch in range(args.max_epochs):
        print(f'starting epoch: {epoch}')
        start = time.time()

        if distributed:
            train_loader.sampler.set_epoch(epoch) #pyright: ignore

        train_loss = train(model, loss_fn, train_loader, optimizer, current_device, args.loss_type, scheduler, PROFILE)
        end = time.time()
        print(f'Epoch: {epoch}, train loss: {train_loss}, time: {(end - start)/60} minutes')

        model.eval()
        with torch.no_grad():
            plot_recon(model, train_loader, current_device, writer, epoch, args.loss_type, training_type='train')
            start = time.time()
            val_loss = validate(model, loss_fn, val_loader, current_device, args.loss_type, PROFILE)
            end = time.time()
            print(f'Epoch: {epoch}, val loss: {val_loss}, time: {(end - start)/60} minutes')
            plot_recon(model, val_loader, current_device, writer, epoch, args.loss_type)

        if current_device == 0:
            if epoch % 20 == 19:
                save_model(os.path.join(writer_dir, 'weight_dir/'), model, optimizer, epoch + 1, current_device)
            if writer:
                writer.add_scalar('train/loss', train_loss, epoch)
                writer.add_scalar('val/loss', val_loss, epoch)
                if scheduler:
                    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

    if current_device == 0:
        save_model(os.path.join(writer_dir, 'weight_dir/'), model, optimizer, args.max_epochs, current_device)


        nmse, ssim, psnr = test(model, test_loader, PROFILE, mask_output=True)
        metrics = {}
        dataset = test_loader.dataset
        print(test_loader)
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        dataset = dataset.dataset
        

        contrast_order = dataset.contrast_order
        all_contrasts = ['t1', 't1ce', 'flair', 't2']
        remaining_contrasts = [contrast for contrast in all_contrasts if contrast not in contrast_order]
        for i in range(len(nmse)):
            metrics['mse-' + contrast_order[i]] = nmse[i]
            metrics['ssim-' + contrast_order[i]] = ssim[i]
            metrics['psnr-' + contrast_order[i]] = psnr[i]

        for contrast in remaining_contrasts:
            metrics['mse-' + contrast] = 0
            metrics['ssim-' + contrast] = 0
            metrics['psnr-' + contrast] = 0
            

        hparams_writer = SummaryWriter('/home/kadotab/scratch/runs/metrics')
        args.contrasts = ','.join(args.contrasts)
        hparams = vars(args)
        print(hparams)
        run_type = writer_dir.split('/')[-2]
        hparams_writer.add_hparams(
                hparams,
                metrics,
                run_name=run_type + run_identifier
                )

    if distributed:
        destroy_process_group()

def create_writer(current_device, args):
    if current_device == 0:
        run_identifier = str(args.R) + '-' + ','.join(args.contrasts) + '-' +  args.loss_type 
        writer_dir = os.path.join(args.log_dir, '') 
        if os.path.exists(os.path.join(writer_dir, run_identifier)):
            while os.path.exists(os.path.join(writer_dir, run_identifier)):
                run_number = re.search(r'\d+$', run_identifier)
                if run_number:
                    parts = re.split(r'\d+$', run_identifier)
                    run_identifier = parts[0] + str(int(run_number.group()) + 1)
                else:
                    run_identifier += str(0)

        writer_dir = os.path.join(writer_dir, run_identifier)

        os.makedirs(os.path.join(writer_dir, 'weight_dir'))
        save_config(args, writer_dir)
        writer = SummaryWriter(writer_dir)
    else:
        writer = None
        run_identifier = ''
        writer_dir = ''
    return writer, writer_dir, run_identifier

def prepare_data(arg: argparse.Namespace, distributed: bool):
    data_dir = arg.data_dir
    
    if arg.dataset == 'brats':
        train_dataset = BratsDataset(os.path.join(data_dir, 'train'), nx=arg.nx, ny=arg.ny, contrasts=arg.contrasts)
        val_dataset = BratsDataset(os.path.join(data_dir, 'val'), nx=arg.nx, ny=arg.ny, contrasts=arg.contrasts)
        test_dataset = BratsDataset(os.path.join(data_dir, 'test'), nx=arg.nx, ny=arg.ny, contrasts=arg.contrasts)
    elif arg.dataset == 'm4raw':
        train_dataset = M4Raw(os.path.join(data_dir, 'train'), nx=arg.nx, ny=arg.ny, contrasts=arg.contrasts)
        val_dataset = M4Raw(os.path.join(data_dir, 'val'), nx=arg.nx, ny=arg.ny, contrasts=arg.contrasts)
        test_dataset = M4Raw(os.path.join(data_dir, 'test'), nx=arg.nx, ny=arg.ny, contrasts=arg.contrasts)


    undersampling_args = {
                'R': arg.R, 
                'R_hat': arg.R_hat, 
                'acs_lines': arg.acs_lines, 
                'poly_order': arg.poly_order,
                'transforms': normalize(arg.norm_method)
            }
    
    train_dataset = UndersampleDecorator(train_dataset, **undersampling_args)
    val_dataset = UndersampleDecorator(val_dataset, **undersampling_args)
    test_dataset = UndersampleDecorator(test_dataset, **undersampling_args)
    
    if arg.use_subset:
        train_dataset, _ = random_split(train_dataset, [0.1, 0.9])
        val_dataset, _ = random_split(val_dataset, [0.1, 0.9])
        test_dataset, _ = random_split(test_dataset, [0.1, 0.9])

    if distributed:
        print('Setting up distributed sampler')
        train_sampler = DistributedSampler(train_dataset, seed=torch.seed())
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        shuffle = False
    else:
        train_sampler = None 
        val_sampler=None
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
                            sampler=val_sampler,
                            num_workers=arg.num_workers, 
                            pin_memory=True,
                            )

    test_loader = DataLoader(test_dataset, 
                            batch_size=arg.batch_size, 
                            num_workers=arg.num_workers,
                            pin_memory=True,
                            )

    return train_loader, val_loader, test_loader



def plot_recon(model, data_loader, device, writer, epoch, loss_type, training_type='val'):
    """ plots a single slice to tensorboard. Plots reconstruction, ground truth, 
    and error magnified by 4

    Args:
        model (nn.Module): model used for reconstruction
        val_loader (nn.utils.DataLoader): dataloader used to get slice
        device (str | int): device number/type ('cpu' or 'gpu')
        writer (torch.utils.SummaryWriter): tensorboard summary writer
        epoch (int): epoch
        loss_type (str): supervised, ssdu, noiser2noise
    """
    if device == 0:

        # get data 
        if training_type == 'val':
            sample = tuple(data.unsqueeze(0) for data in data_loader.dataset[10])
        elif training_type == 'train':
            sample = next(iter(data_loader))
        else:
            raise ValueError(f'type should be either val or test but got {training_type}')

        # forward pass
        mask, input_slice, target_slice, loss_mask, zf_mask = to_device(sample, device, 'supervised')
        output = model(input_slice, mask)
        output *= zf_mask
        output = output * (input_slice == 0) + input_slice
        output = output.cpu()
        
        # plot sensetivity maps
        cur_model = model
        if isinstance(cur_model, torch.nn.parallel.DistributedDataParallel):
            cur_model = cur_model.module
        sensetivity_maps = cur_model.sens_model(input_slice, mask)
        for i in range(sensetivity_maps.shape[1]):
            writer.add_images(training_type + '-sense_map/image_' + str(i), sensetivity_maps[0, i, :, :, :].cpu().abs().unsqueeze(1), epoch)

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

        # difference magnitude
        difference_scaling = 4
        # scale images and difference
        image_scaled = output.abs()/image_scaling_factor
        diff_scaled = diff/(image_scaling_factor/difference_scaling)
        input_scaled = x_input.abs()/(image_scaling_factor)

        # clamp to 0-1 range
        image_scaled = image_scaled.clamp(0, 1)
        diff_scaled = diff_scaled.clamp(0, 1)
        input_scaled = input_scaled.clamp(0, 1)

        writer.add_images('images/' + training_type + '/recon', image_scaled.unsqueeze(1), epoch)
        writer.add_images('images/' + training_type + '/diff', diff_scaled.unsqueeze(1), epoch)
        writer.add_images('images/' + training_type + '/input', input_scaled.unsqueeze(1), epoch)

        if loss_type != 'supervised':
            mask_lambda_omega, _, _, loss_mask, zf_mask = to_device(sample, device, loss_type)
            writer.add_images('mask/' + training_type + '/omega_lambda_mask', mask_lambda_omega[0, :, [0], :, :], epoch)
            writer.add_images('mask/' + training_type + '/omega_mask', mask[0, :, [0], :, :], epoch)
            writer.add_images('mask/' + training_type + '/omega_not_lambda', loss_mask[0, :, [0], :, :].float(), epoch)
        else:
            writer.add_images('mask/' + training_type + '/inital_mask', mask[0, :, [0], :, :], epoch)
            writer.add_images('mask/' + training_type + '/loss_mask', loss_mask[0, :, [0], :, :], epoch)

            
        # plot target if it's the first epcoh
        recon_scaled = ground_truth/image_scaling_factor
        recon_scaled = recon_scaled.clamp(0, 1)
        writer.add_images('images/' + training_type + '/target', recon_scaled.unsqueeze(1).abs(), epoch)


def setup_model_backbone(args):
    model_name = args.model
    cascades = args.cascades
    input_channels = 2*len(args.contrasts)
    if model_name == 'unet':
        backbone = partial(Unet, in_chan=input_channels, out_chan=input_channels, depth=4, chans=args.channels)
    elif model_name == 'resnet':
        backbone = partial(ResNet, in_chan=input_channels, out_chan=input_channels, itterations=15, chans=args.channels)
    elif model_name == 'dncnn':
        backbone = partial(DnCNN, in_chan=input_channels, out_chan=input_channels, channels=args.channels, num_of_layers=15)
    elif model_name == 'transformer':
        backbone = partial(SwinUNETR, img_size=(128, 128), in_channels=2, out_channels=2, spatial_dims=2, feature_size=12)
        print('loaded swinunet!')
    elif model_name == 'unetr':
        backbone = partial(UnetR, 
                           in_chan=input_channels, 
                           out_chan=input_channels, 
                           img_size=256, 
                           hidden_size=args.hidden_size,
                           mlp_dim=args.mlp_dim,
                           num_heads=args.num_heads,
                           feature_size=args.feature_size,
                           )
    elif model_name == 'multitask':
        backbone = partial(MultiTaskUnet, 
                           in_chan=input_channels, 
                           out_chan=input_channels, 
                           initial_channel=args.initial_chan,
                           initial_depth=args.initial_depth,
                           joint_depth=args.joint_depth,
                           joint_channel=args.joint_chan
                           )
        print('loaded multi-task!')
    else:
        raise ValueError(f'Backbone should be either unet resnet or dncnn but found {model_name}')

    model = VarNet_mc(backbone, num_cascades=cascades)
    params = sum([x.numel()  for x in model.parameters()])
    print(f'Model has {params:,}')

    return model


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
    parser.add_argument('--model', type=str, choices=['unet', 'resnet', 'dncnn', 'transformer', 'multitask', 'unetr'], default='unet')
    parser.add_argument('--loss_type', type=str, choices=['supervised', 'noiser2noise', 'ssdu', 'k-weighted'], default='ssdu')
    parser.add_argument('--scheduler', type=str, choices=['none', 'cyclic', 'cosine_anneal', 'steplr'], default='none')
    parser.add_argument('--channels', type=int, default=18, help='')
    parser.add_argument('--cascades', type=int, default=6, help='')
    parser.add_argument('--weight_decay', type=float, default=0, help='')
    parser.add_argument('--norm_method', type=str, choices=['mean', 'max', 'std', 'mean2', 'k'], default='mean', help='')
    parser.add_argument('--log_dir', type=str, default='/home/kadotab/scratch/runs/', help='')
    
    parser.add_argument('--init_method', default='tcp://localhost:18888', type=str, help='')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='')
    parser.add_argument('--world_size', default=2, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    parser.add_argument('--use_subset', action='store_true', help='')
    parser.add_argument('--dataset', default='brats', type=str, help='')
    parser = BratsDataset.add_model_specific_args(parser)
    parser = UndersampleDecorator.add_model_specific_args(parser)

    args, _ = parser.parse_known_args()

    if args.model == 'unet':
        parser = Unet.add_model_specific_args(parser)
    elif args.model == 'resnet':
        parser = ResNet.add_model_specific_args(parser)
    elif args.model == 'multitask':
        parser = MultiTaskUnet.add_model_specific_args(parser)
    elif args.model == 'unetr':
        parser = UnetR.add_model_specific_args(parser)

    args = parser.parse_args()
    
    main(args)
