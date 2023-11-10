import os
import contextlib
import json
from functools import partial
import torch
from typing import Union

import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_scheduler(train_loader, optimizer, scheduler_type) -> Union[torch.optim.lr_scheduler.LRScheduler, None]:
    if scheduler_type == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader)*10, T_mult=1, eta_min=1e-5)
    elif scheduler_type == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 5e-5, 1e-3, len(train_loader)*8, mode='triangular2', cycle_momentum=False)
    elif scheduler_type == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, len(train_loader)*50)
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
    if profile:
        print('PROFILING')

    running_loss = torch.Tensor([0]).to(device)
    cm = setup_profile_context_manager(profile, 'train')
 
    with cm as prof:
        for step, data in enumerate(dataloader):
            
            running_loss += train_step(model, loss_function, optimizer, data, device, loss_type)

            if prof:
                print('PROF STEP')
                prof.step()
                if step >= (1 + 1 + 10) * 2:
                    break

            if scheduler:
                scheduler.step()


    return running_loss.item()/len(dataloader)

def setup_profile_context_manager(profile, train_type):
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
    
    if profile:
        context_manager = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/kadotab/scratch/runs/profile_' + train_type + '-' + str(max_files_number + 1)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
            ) 
    else:
        context_manager = contextlib.nullcontext()
    
    return context_manager


def train_step(model, loss_function, optimizer, data, device, loss_type) -> torch.Tensor:
    mask, input_slice, target_slice, loss_mask, zf_mask  = to_device(data, device, loss_type)

    optimizer.zero_grad()
    predicted_sampled = model(input_slice, mask)
    predicted_sampled *= zf_mask

    loss = loss_function(
            torch.view_as_real(target_slice * loss_mask),
            torch.view_as_real(predicted_sampled * loss_mask)
            )
    
    # normalize to number of ssdu indecies not number of voxels
    #loss = loss * predicted_sampled.numel() / loss_mask.sum()

    loss.backward()
    optimizer.step()
    loss_step = loss.detach() * target_slice.shape[0]
    return loss_step


def validate(model, loss_function, dataloader, device, supervised, profile=False):
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

    cm = setup_profile_context_manager(profile, 'val')
    val_running_loss = torch.Tensor([0]).to(device)
    with cm as prof:
        for step, data in enumerate(dataloader):
            if prof:
                print('PROF STEP')
                prof.step()
                if step >= (1 + 1 + 10) * 2:
                    break
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
            torch.view_as_real(target_slice * loss_mask),
            torch.view_as_real(predicted_sampled * loss_mask)
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
    double_undersaple, undersample, k_space, k, omega_mask, lambda_mask = data
    zero_fill_mask = k_space != 0
    if loss_type == 'supervised':
        input_slice = undersample.to(device)
        target_slice = k_space.to(device)
        loss_mask = torch.ones_like(target_slice, dtype=torch.bool)
        mask = omega_mask
    else:
        input_slice = double_undersaple.to(device)
        target_slice = undersample.to(device)
        mask = omega_mask * lambda_mask

        if loss_type == 'nosier2noise':
            loss_mask = omega_mask
        elif loss_type == 'ssdu' or loss_type == 'k-weighted':
            loss_mask = omega_mask & ~lambda_mask
        else:
            raise ValueError(f'loss mask should be ssdu, noiser2noise, k-weighted, or supervised but got {loss_type}')

        if loss_type == 'k-weighted' or loss_type == 'nosier2noise':
            loss_mask = loss_mask.type(torch.float32)
            loss_mask /= torch.sqrt(1 - k.unsqueeze(1))
            
    zero_fill_mask = zero_fill_mask.to(device)
    mask = mask.to(device)
    loss_mask = loss_mask.to(device)

    return mask, input_slice, target_slice, loss_mask, zero_fill_mask 


def save_config(args, writer_dir):
    args_dict = vars(args)
    with open(os.path.join(writer_dir, 'config.txt'), 'w') as f:
        f.write(json.dumps(args_dict, indent=4))

    
