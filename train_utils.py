import os
import contextlib
import json
from functools import partial
import torch
from typing import Union

import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from ml_recon.models.varnet import VarNet
from ml_recon.models import Unet, ResNet, DnCNN, SwinUNETR

def setup_scheduler(train_loader, optimizer, scheduler_type) -> Union[torch.optim.lr_scheduler.LRScheduler, None]:
    if scheduler_type == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader)*10, T_mult=1, eta_min=1e-4)
    elif scheduler_type == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 5e-5, 1e-3, len(train_loader)*8, mode='triangular2', cycle_momentum=False)
    elif scheduler_type == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, len(train_loader)*200)
    elif scheduler_type == 'none':
        scheduler = None
    else:
        scheduler = None
    return scheduler

def setup_ddp(current_device, is_distributed, model):
    if is_distributed:
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
    is_distributed = False
    if ngpus_per_node > 1:
        # multiple gpu training
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
        distributed.init_process_group(backend=dist_backend, init_method=init_method, world_size=world_size, rank=current_device)
        print("process group ready!")
        is_distributed = True
    elif ngpus_per_node == 1:
        # one gpu training
        current_device = 0
        is_distributed = False
    else:
        #cpu training
        current_device = 'cpu'
        is_distributed = False

    return current_device, is_distributed


def train(model, loss_function, dataloader, optimizer, device, loss_type, scheduler, profile=False):
    running_loss = torch.Tensor([0]).to(device)

    cm = setup_profile_context_manager(profile)
 
    with cm as prof:
        for i, data in enumerate(dataloader):
            if profile:
                print('PROFILING')
                if i > (1 + 1 + 3) * 2:
                    break
                if prof:
                    prof.step()

            running_loss += train_step(model, loss_function, optimizer, data, device, loss_type)
            if scheduler:
                scheduler.step()

    return running_loss.item()/len(dataloader)

def setup_profile_context_manager(profile):
    """ create a context manager if global PROFILE flag is set to true else retruns
    null context manager

    Returns:
        context manager: returns profile context manager
    """
    prof_files = [file for file in os.listdir('/home/kadotab/scratch/runs/') if 'profile' in file]
    file_numbers = [int(file.split('-')[-1]) for file in prof_files]
    if file_numbers:
        max_files_number = max(file_numbers)
    else: 
        max_files_number = 0
    
    context_manager = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/kadotab/scratch/runs/profile-' + str(max_files_number + 1)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) if profile else contextlib.nullcontext()
    
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
    #loss = loss * predicted_sampled.numel() / loss_mask.sum()

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

    #loss = loss * predicted_sampled.numel() / loss_mask.sum()

    return loss.detach() * target_slice.shape[0]


def to_device(data: tuple, device: str, loss_type: str):
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


def save_config(args, writer_dir):
    args_dict = vars(args)
    with open(os.path.join(writer_dir, 'config.txt'), 'w') as f:
        f.write(json.dumps(args_dict, indent=4))

    
