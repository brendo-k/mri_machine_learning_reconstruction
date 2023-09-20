from datetime import datetime
from typing import Dict
import os
import argparse
import time
import yaml
import contextlib
import json
from functools import partial
import torch
from typing import Union

from ml_recon.models.varnet import VarNet
from ml_recon.models import Unet, ResNet, DnCNN, SwinUNETR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from ml_recon.transforms import normalize
from ml_recon.dataset.sliceloader import SliceDataset 
from ml_recon.utils import save_model, ifft_2d_img, root_sum_of_squares

from torchvision.transforms import Compose
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Globals
PROFILE = False

# Argparse


def main(args):

    current_device, distributed = setup_devices(args.dist_backend, args.init_method, args.world_size)

    model = setup_model_backbone(args.model, current_device)

    model = setup_ddp(current_device, distributed, model)

    train_loader, val_loader = prepare_data(args, distributed)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer_dir = '/home/kadotab/scratch/runs/' + datetime.now().strftime("%m%d-%H:%M:%S") + model.__class__.__name__ + '-' + args.model + '-' + args.loss_type

    try:
        os.makedirs(os.path.join(writer_dir, 'model_weights'))
    except FileExistsError as e:
        print(f'Found {e}, creating new directory')
        os.makedirs(os.path.join(writer_dir + 2, 'model_weights'))

        
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
            if writer:
                plot_recon(model, train_loader, current_device, writer, epoch, args.loss_type, type='train')
                plot_recon(model, val_loader, current_device, writer, epoch, args.loss_type)
            val_loss = validate(model, loss_fn, val_loader, current_device, args.loss_type)
        
        if current_device == 0:
            if writer:
                writer.add_scalar('train/loss', train_loss, epoch)
                writer.add_scalar('val/loss', val_loss, epoch)
                if scheduler:
                    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
        if epoch % 24 == 0: 
            save_model(os.path.join(writer_dir, 'model_weights'), model, optimizer, args.max_epochs, current_device)

    save_model(os.path.join(writer_dir, 'model_weights'), model, optimizer, args.max_epochs, current_device)

    if distributed:
        destroy_process_group()

def setup_scheduler(train_loader, optimizer, scheduler_type) -> Union[torch.optim.lr_scheduler._LRScheduler, None]:
    if scheduler_type == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader)*10, T_mult=1, eta_min=1e-4)
    elif scheduler_type == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 5e-5, 1e-3, len(train_loader)*8, mode='triangular2', cycle_momentum=False)
    elif scheduler_type == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, len(train_loader)*200)
    elif scheduler_type == 'none':
        scheduler = None
    return scheduler

def setup_ddp(current_device, distributed, model):
    if distributed:
        print(f'Setting up DDP in device {current_device}')
        model = DDP(model, device_ids=[current_device])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def setup_model_backbone(model_name, current_device, in_chan=2, out_chan=2):
    if model_name == 'unet':
        backbone = partial(Unet, in_chan=in_chan, out_chan=out_chan, depth=4, chans=18)
    elif model_name == 'resnet':
        backbone = partial(ResNet, in_chan=in_chan, out_chan=out_chan, itterations=15, chans=32)
    elif model_name == 'dncnn':
        backbone = partial(DnCNN, in_chan=in_chan, out_chan=out_chan, feature_size=32, num_of_layers=15)
    elif model_name == 'transformer':
        backbone = partial(SwinUNETR, img_size=(128, 128), in_channels=2, out_channels=2, spatial_dims=2, feature_size=12)
        print('loaded swinunet!')
    else:
        raise ValueError(f'Backbone should be either unet resnet or dncnn but found {model_name}')

    model = VarNet(backbone, num_cascades=6)
    params = sum([x.numel()  for x in model.parameters()])
    print(f'Model has {params:,}')
    model.to(current_device)

    return model

def setup_devices(dist_backend, init_method, world_size):
    """set up ddp, gpu, or cpu based on arguments and current number of devices 

    Args:
        dist_backend (str): string of backend used
        init_method (str): string of server address and port
        world_size (int): max num of gpus 
    """    

    ngpus_per_node = torch.cuda.device_count()
    current_device = None
    distributed = False
    if ngpus_per_node > 1:
        print("Starting DPP...")

        """ This next line is the key to getting DistributedDataParallel working on SLURM:
		SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
 		current process inside a node and is also 0 or 1 in this example."""

        slurm_id = os.environ.get("SLURM_LOCALID")
        if not slurm_id:
            raise ValueError('Did not get a slurm id')
        
        current_device = int(slurm_id) 

        """ this block initializes a process group and initiate communications
		be  tween all processes running on all nodes """

        print(f'From Rank: {current_device}, ==> Initializing Process Group...')
        #init the process group
        init_process_group(backend=dist_backend, init_method=init_method, world_size=world_size, rank=current_device)
        print("process group ready!")
        distributed = True
    elif ngpus_per_node == 1:
        current_device = 0
        distributed = False
    else:
        current_device = 'cpu'
        distributed = False
    return current_device, distributed


def prepare_data(arg, distributed: bool):
    transforms = Compose(
        (
            normalize(),
        )
    )
    data_dir = arg.data_dir

    dataset_args = {
            'nx': arg.nx,
            'ny': arg.ny,  
            'R': arg.R,
            'R_hat': arg.R_hat,
            'poly_order': arg.poly_order,
            }
    
    train_dataset = SliceDataset(os.path.join(data_dir, 'multicoil_train'),
                                 transforms=transforms,
                                 **dataset_args
                                 )
    
    val_dataset = SliceDataset(os.path.join(data_dir, 'multicoil_val'),
                                 transforms=transforms,
                                 **dataset_args
                                 )
    
    if arg.use_subset:
        train_dataset, _ = random_split(train_dataset, [0.1, 0.9])

    if distributed:
        print('Setting up distributed sampler')
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        shuffle = False
    else:
        train_sampler, val_sampler = None, None
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


def train(model, loss_function, dataloader, optimizer, device, loss_type, scheduler):
    running_loss = torch.Tensor([0]).to(device)

    cm = setup_profile_context_manager()
 
    with cm as prof:
        for i, data in enumerate(dataloader):
            if PROFILE:
                print('PROFILING')
                if i > (1 + 1 + 3) * 2:
                    break
                if prof:
                    prof.step()

            running_loss += train_step(model, loss_function, optimizer, data, device, loss_type)
            if scheduler:
                scheduler.step()

    return running_loss.item()/len(dataloader)

def setup_profile_context_manager():
    """ create a context manager if global PROFILE flag is set to true else retruns
    null context manager

    Returns:
        context manager: returns profile context manager
    """
    context_manager = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/kadotab/scratch/runs/varnet_batch0_workers1'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) if PROFILE else contextlib.nullcontext()
    
    return context_manager


def train_step(model, loss_function, optimizer, data, device, loss_type) -> torch.Tensor:
    mask, input_slice, target_slice, loss_mask, zf_mask  = to_device(data, device, loss_type)

    optimizer.zero_grad()
    predicted_sampled = model(input_slice, mask)
    predicted_sampled *= zf_mask

    loss = loss_function(
            torch.view_as_real(predicted_sampled * loss_mask),
            torch.view_as_real(target_slice * loss_mask)
            )
    
    # normalize to number of ssdu indecies not number of voxels
    loss = loss * predicted_sampled.numel() / loss_mask.sum()

    loss.backward()
    optimizer.step()
    loss_step = loss.detach() * target_slice.shape[0]
    return loss_step


def validate(model, loss_function, dataloader, device, supervised):
    """ Validation loop. Loops through the entire validation dataset

    Args:
        model (nn.Module): model being trained
        loss_function (nn.LossFunction): loss function to be used
        dataloader (nn.utils.DataLoader): validation dataloader to be used
        device (str): device name (gpu, cpu)
        supervised (boolean): boolean flag to set supervised or unsupervised

    Returns:
        int: average validation loss per sample
    """
    val_running_loss = torch.Tensor([0]).to(device)
    for data in dataloader:
        val_running_loss += val_step(model, loss_function, device, data, supervised)
    return val_running_loss.item()/len(dataloader)


def val_step(model, loss_function, device, data, supervised):
    """ validation step. Performs a validation step for a single mini-batch

    Args:
        model (nn.Module): model being validated
        loss_function (nn.LossFunction): loss function used as metric
        device (str): device name (gpu, cpu)
        data (torch.Tensor): tensor of the current mini-batch
        supervised (bool): boolean flag to set supervised or unsupervised learning

    Returns:
        torch.Tensor[float]: loss of invdividual sample (mini-batch * num_batches)
    """
    mask, input_slice, target_slice, loss_mask, zf_mask = to_device(data, device, supervised)

    predicted_sampled = model(input_slice,  mask)
    predicted_sampled *= zf_mask
    loss = loss_function(
            torch.view_as_real(predicted_sampled * loss_mask),
            torch.view_as_real(target_slice * loss_mask)
            )

    loss = loss * predicted_sampled.numel() / loss_mask.sum()

    return loss.detach() * target_slice.shape[0]


def to_device(data: Dict, device: str, loss_type: str):
    """ moves tensors in data to the device specified

    Args:
        data (Dict): data dictionary returned from dataloader
        device (str): device to move data onto
        supervised (bool): return supervised data if true

    Returns:
        torch.Tensor: returns multiple tensors now on the device
    """
    double_undersaple, undersample, k_space, k = data
    zero_fill_mask = k_space != 0
    if loss_type == 'supervised':
        input_slice = undersample
        target_slice = k_space
        mask = undersample != 0
        loss_mask = torch.ones_like(mask)
    else:
        input_slice = double_undersaple
        target_slice = undersample

        if loss_type == 'nosier2noise':
            loss_mask = target_slice != 0
            mask = input_slice != 0
        elif loss_type == 'ssdu' or loss_type == 'k-weighted':
            loss_mask = (target_slice != 0) & (input_slice == 0)
            mask = input_slice != 0
        else:
            raise ValueError(f'loss mask should be ssdu, noiser2noise, k-weighted, or supervised but got {loss_type}')

        if loss_type == 'k-weighted' or loss_type == 'nosier2noise':
            loss_mask = loss_mask.type(torch.float32)
            loss_mask /= torch.sqrt(1 - k.unsqueeze(1))
            
    input_slice = input_slice.to(device)
    target_slice = target_slice.to(device)
    mask = mask.to(device)
    loss_mask = loss_mask.to(device)
    zero_fill_mask = zero_fill_mask.to(device)

    return mask, input_slice, target_slice, loss_mask, zero_fill_mask 


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
        output = root_sum_of_squares(ifft_2d_img(output), coil_dim=1).cpu()
        ground_truth = root_sum_of_squares(ifft_2d_img(target_slice), coil_dim=1).cpu()
        x_input = root_sum_of_squares(ifft_2d_img(input_slice), coil_dim=1).cpu()

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
    with open(args.config, 'r') as f:
        configs_dict = yaml.load(f)
        
    for k, v in configs_dict.items():
        setattr(args, k, v)

    return args

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Varnet self supervised trainer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate to use')
    parser.add_argument('--batch_size', type=int, default=5, help='')
    parser.add_argument('--max_epochs', type=int, default=50, help='')
    parser.add_argument('--num_workers', type=int, default=0, help='')
    parser.add_argument('--model', type=str, choices=['unet', 'resnet', 'dncnn', 'transformer'], default='unet')
    parser.add_argument('--loss_type', type=str, choices=['supervised', 'noiser2noise', 'ssdu', 'k-weighted'], default='ssdu')
    parser.add_argument('--scheduler', type=str, choices=['none', 'cyclic', 'cosine_anneal', 'steplr'], default='none')
    parser.add_argument('--use_subset', action='store_true', help='')
    parser.add_argument('--config', default='', help='')
    
    parser.add_argument('--init_method', default='tcp://localhost:18888', type=str, help='')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='')
    parser.add_argument('--world_size', default=2, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')

    parser = SliceDataset.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
