from datetime import datetime
from typing import Dict
import os
import argparse
import time
import yaml
import contextlib
import matplotlib.pyplot as plt

from ml_recon.models.varnet_unet import VarNet
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.tensorboard import SummaryWriter

from ml_recon.transforms import toTensor, normalize, pad_recon, pad
from ml_recon.dataset.self_supervised_slice_loader import SelfSupervisedSampling
from ml_recon.utils import save_model, ifft_2d_img
from ml_recon.utils.collate_function import collate_fn

from torchvision.transforms import Compose
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Globals
PROFILE = False

# Argparse
parser = argparse.ArgumentParser(description='Varnet self supervised trainer')
parser.add_argument('--lr', type=float, default=1e-4, help='')
parser.add_argument('--batch_size', type=int, default=2, help='')
parser.add_argument('--max_epochs', type=int, default=50, help='')
parser.add_argument('--num_workers', type=int, default=1, help='')

parser.add_argument('--init_method', default='tcp://localhost:18888', type=str, help='')
parser.add_argument('--dist_backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=2, type=int, help='')
parser.add_argument('--distributed', action='store_false', help='')

def main():
    args = parser.parse_args()
    
    current_device, distributed = setup_devices(args)

    model = VarNet(num_cascades=5)
    model.to(current_device)

    if distributed:
        print(f'Setting up DDP in device {current_device}')
        model = DDP(model, device_ids=[current_device])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    train_loader, val_loader = prepare_data(args, distributed)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if current_device == 0:
        writer_dir = '/home/kadotab/scratch/runs/' + datetime.now().strftime("%m%d-%H%M") + model.__class__.__name__
        writer = SummaryWriter(writer_dir)

    path = '/home/kadotab/python/ml/ml_recon/Model_Weights/'
    lr = torch.linspace(1e-8, 0.1, 200)
    losses = []
    sample = next(iter(train_loader))
    mask, undersampled_slice, sampled_slice, ssdu_indecies = to_device(sample, current_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for lrs in lr:
        optimizer = torch.optim.Adam(model.parameters(), lr=lrs)
        output = model(undersampled_slice, mask)
        loss = loss_fn(torch.view_as_real(output * ssdu_indecies), torch.view_as_real(sampled_slice * ssdu_indecies))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    plt.plot(lr, losses)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('/home/kadotab/python/ml/loss_vs_lr.png')

   

    if distributed:
        destroy_process_group()

def setup_devices(args):
    """set up 

    Args:
        args (_type_): _description_
    """    

    ngpus_per_node = torch.cuda.device_count()
    current_device = None
    distributed = False
    if ngpus_per_node > 1:
        print("Starting DPP...")

        """ This next line is the key to getting DistributedDataParallel working on SLURM:
		SL  URM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
 		c   urrent process inside a node and is also 0 or 1 in this example."""

        current_device = int(os.environ.get("SLURM_LOCALID")) 

        """ this block initializes a process group and initiate communications
		be  tween all processes running on all nodes """

        print(f'From Rank: {current_device}, ==> Initializing Process Group...')
        #init the process group
        init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=current_device)
        print("process group ready!")
        distributed = True
    elif ngpus_per_node == 1:
        current_device = 0
        distributed = False
    else:
        current_device = 'cpu'
        distributed = False
    return current_device, distributed


def prepare_data(arg: argparse.ArgumentParser, distributed: bool):
    transforms = Compose(
        (
            pad((640, 320)),
            pad_recon((320, 320)),
            toTensor(),
            normalize(),
        )
    )
    train_dataset = SelfSupervisedSampling(
        '/home/kadotab/train.json',
        transforms=transforms,
        R=4,
        R_hat=2
        )
    
    val_dataset = SelfSupervisedSampling(
        '/home/kadotab/val.json',
        transforms=transforms,
        R=4,
        R_hat=2
        )

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
                              collate_fn=collate_fn
                              )

    val_loader = DataLoader(val_dataset, 
                            batch_size=arg.batch_size, 
                            num_workers=arg.num_workers, 
                            collate_fn=collate_fn,
                            sampler=val_sampler,
                            )
    return train_loader, val_loader



def train(model, loss_function, dataloader, optimizer, device):
    running_loss = 0

    cm = setup_profile_context_manager()
 
    with cm as prof:
        for i, data in enumerate(dataloader):
            if PROFILE:
                if i > (1 + 1 + 3) * 2:
                    break
                prof.step()

            running_loss += train_step(model, loss_function, optimizer, data, device)

    return running_loss/len(dataloader)

def setup_profile_context_manager():
    context_manager = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/kadotab/scratch/runs/varnet_batch1_workers0'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) if PROFILE else contextlib.nullcontext()
    
    return context_manager


def train_step(model, loss_function, optimizer, data, device):
    mask, undersampled_slice, sampled_slice, ssdu_indecies = to_device(data, device)

    optimizer.zero_grad()
    predicted_sampled = model(undersampled_slice, mask)
    loss = loss_function(
            torch.view_as_real(predicted_sampled * ssdu_indecies),
            torch.view_as_real(sampled_slice * ssdu_indecies)
            )
    # normalize to number of ssdu indecies not number of voxels
    loss = loss * predicted_sampled.numel() / ssdu_indecies.sum()

    loss.backward()
    optimizer.step()
    loss_step = loss.item()*sampled_slice.shape[0]
    return loss_step


def validate(model, loss_function, dataloader, device):
    """ Validation loop. Loops through the entire validation dataset

    Args:
        model (nn.Module): model being trained
        loss_function (nn.LossFunction): loss function to be used
        dataloader (nn.utils.DataLoader): validation dataloader to be used
        device (str): device name (gpu, cpu)

    Returns:
        int: average validation loss per sample
    """
    val_running_loss = 0
    for data in dataloader:
        val_running_loss += val_step(model, loss_function, device, data)
    return val_running_loss/len(dataloader)


def val_step(model, loss_function, device, data):
    """ validation step. Performs a validation step for a single mini-batch

    Args:
        model (nn.Module): model being validated
        loss_function (nn.LossFunction): loss function used as metric
        device (str): device name (gpu, cpu)
        data (torch.Tensor): tensor of the current mini-batch

    Returns:
        torch.Tensor[float]: loss of invdividual sample (mini-batch * num_batches)
    """
    mask, undersampled_slice, sampled_slice, ssdu_indecies = to_device(data, device)

    predicted_sampled = model(undersampled_slice,  mask)
    loss = loss_function(
            torch.view_as_real(predicted_sampled * ssdu_indecies),
            torch.view_as_real(sampled_slice * ssdu_indecies)
            )

    loss = loss * predicted_sampled.numel() / ssdu_indecies.sum()

    return loss.item()*sampled_slice.shape[0]


def to_device(data: Dict, device: str):
    """ moves tensors in data to the device specified

    Args:
        data (Dict): data dictionary returned from dataloader
        device (str): device to move data onto

    Returns:
        torch.Tensor: returns multiple tensors now on the device
    """
    undersampled = data['undersampled']
    mask_lambda = data['omega_mask']
    mask = data['mask']
    double_undersampled = data['double_undersample']
    K = data['k']

    undersampled_slice = double_undersampled.to(device)
    sampled_slice = undersampled.to(device)
    set_1_mask = (mask_lambda * mask).to(device)

    ssdu_indecies = (~mask_lambda * mask).detach()
    ssdu_indecies = ssdu_indecies[:, None, :, :].repeat(1, undersampled_slice.shape[1], 1, 1).to(device)

    return set_1_mask, undersampled_slice, sampled_slice, ssdu_indecies


def plot_recon(model, val_loader, device, writer, epoch):
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
        sample = next(iter(val_loader))
        sampled = sample['undersampled']
        output = model(sampled.to(device), sample['mask'].to(device))
        output = output * sample['scaling_factor'].to(device)[:, None, None, None]
        output = ifft_2d_img(output)
        output = output.abs().pow(2).sum(1).sqrt()
        output = output[:, 160:-160, :].cpu()
        diff = (output - sample['recon']).abs()

        image_scaling_factor = sample['recon'][0].abs().max() * 0.50
        image_scaled = output[0].abs().unsqueeze(0)/image_scaling_factor
        image_scaled[image_scaled > 1] = 1

        diff_scaled = diff[0].abs().unsqueeze(0)/(image_scaling_factor/4)
        diff_scaled[diff_scaled > 1] = 1

        writer.add_image('val/recon', image_scaled, epoch)
        writer.add_image('val/diff', diff_scaled, epoch)

        if epoch == 0:
            recon_scaled = sample['recon'][0].abs().unsqueeze(0)/image_scaling_factor
            recon_scaled[recon_scaled> 1] = 1
            writer.add_image('val/target', recon_scaled, epoch)


def load_config(cname):
    """ loads yaml file """
    with open(cname, 'r') as stream:
        configs = yaml.safe_load_all(stream)
        config = next(configs)

    return config

if __name__ == '__main__':
    main()