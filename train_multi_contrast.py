from datetime import datetime
import os
import argparse
import time
import yaml
import json
import torch

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from ml_recon.transforms import normalize
from ml_recon.dataset.multicontrast_loader import MultiContrastLoader 
from ml_recon.utils import save_model, ifft_2d_img, root_sum_of_squares
from torch.distributed import destroy_process_group

from torchvision.transforms import Compose
from torch.utils.data.distributed import DistributedSampler
from train_varnet_self_supervised import (
        to_device, 
        setup_model_backbone,
        setup_devices,
        setup_scheduler,
        setup_ddp,
        train,
        validate, 
        val_step,
        to_device
        )


# Globals
PROFILE = False

# Argparse
parser = argparse.ArgumentParser(description='Varnet self supervised trainer')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate to use')
parser.add_argument('--batch_size', type=int, default=5, help='')
parser.add_argument('--max_epochs', type=int, default=50, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--data_dir', type=str, default='/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/training_subset', help='')
parser.add_argument('--model', type=str, choices=['unet', 'resnet', 'dncnn', 'transformer'], default='unet')
parser.add_argument('--loss_type', type=str, choices=['supervised', 'noiser2noise', 'ssdu', 'k-weighted'], default='ssdu')
parser.add_argument('--scheduler', type=str, choices=['none', 'cyclic', 'cosine_anneal', 'steplr'], default='none')
parser.add_argument('--R_hat', type=float, default=2)

parser.add_argument('--init_method', default='tcp://localhost:18888', type=str, help='')
parser.add_argument('--dist_backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=2, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
parser.add_argument('--use_subset', action='store_true', help='')


def main():
    args = parser.parse_args()
    #torch.manual_seed(0)
    #np.random.seed(0)
    #torch.cuda.manual_seed(0)

    current_device, distributed = setup_devices(args.dist_backend, args.init_method, args.world_size)

    model = setup_model_backbone(args.model, current_device)

    model = setup_ddp(current_device, distributed, model)

    train_loader = prepare_data(args, distributed)

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
        #    val_loss = validate(model, loss_fn, val_loader, current_device, args.loss_type)
        #    plot_recon(model, val_loader, current_device, writer, epoch, args.loss_type)
        #
        
        if current_device == 0:
            if writer:
                writer.add_scalar('train/loss', train_loss, epoch)
                #writer.add_scalar('val/loss', val_loss, epoch)
                if scheduler:
                    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

    save_model(os.path.join(writer_dir, 'weight_dir'), model, optimizer, args.max_epochs, current_device)

    if distributed:
        destroy_process_group()


def prepare_data(arg: argparse.Namespace, distributed: bool):
    transforms = Compose(
        (
            normalize(),
        )
    )
    data_dir = arg.data_dir
    
        
    train_dataset = MultiContrastLoader(os.path.join(data_dir),
                                 transforms=transforms
                                 )
    
    #val_dataset = MultiContrastLoader(os.path.join(data_dir, 'multicoil_val'),
    #                             undersampling=undersampling,
    #                             transforms=transforms
    #                             )
    #
    if arg.use_subset:
        train_dataset, _ = random_split(train_dataset, [0.1, 0.9])

    if distributed:
        print('Setting up distributed sampler')
        train_sampler = DistributedSampler(train_dataset)
        #val_sampler = DistributedSampler(val_dataset)
        shuffle = False
    else:
        #val_sampler = None
        train_sampler = None 
        shuffle = True

    train_loader = DataLoader(train_dataset, 
                              batch_size=arg.batch_size,
                              num_workers=arg.num_workers,
                              shuffle=shuffle, 
                              sampler=train_sampler,
                              pin_memory=True,
                              )

    #val_loader = DataLoader(val_dataset, 
    #                        batch_size=arg.batch_size, 
    #                        num_workers=arg.num_workers, 
    #                        sampler=val_sampler,
    #                        pin_memory=True,
    #                        )
    return train_loader#, val_loader



def plot_recon(model, val_loader, device, writer, epoch, supervised, type='val'):
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
        
        # coil combination
        output = root_sum_of_squares(ifft_2d_img(output), coil_dim=2).cpu()
        ground_truth = root_sum_of_squares(ifft_2d_img(target_slice), coil_dim=2).cpu()
        x_input = root_sum_of_squares(ifft_2d_img(input_slice), coil_dim=2).cpu()
        output = output[0]
        ground_truth = ground_truth[0]
        x_input = x_input[0]

        diff = (output - ground_truth).abs()

        # get scaling factor (skull is high intensity)
        image_scaling_factor = ground_truth.amax((1, 2)).unsqueeze(1).unsqueeze(1) * 0.60

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

        if supervised != 'supervised':
            mask_lambda_omega, _, _, loss_mask, zf_mask = to_device(sample, device, supervised)
            writer.add_images('mask/' + type + '/omega_lambda_mask', mask_lambda_omega[:, [0], :, :], epoch)
            writer.add_images('mask/' + type + '/omega_mask', mask[:, [0], :, :], epoch)
            writer.add_images('mask/' + type + '/omega_not_lambda', loss_mask[:, [0], :, :], epoch)
            
        # plot target if it's the first epcoh
        recon_scaled = ground_truth/image_scaling_factor
        recon_scaled = recon_scaled.clamp(0, 1)
        writer.add_images('images/' + type + '/target', recon_scaled.unsqueeze(1).abs(), epoch)


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
    main()
