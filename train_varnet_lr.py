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

from ml_recon.transforms import to_tensor, normalize, pad_recon, pad, normalize_mean
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
parser.add_argument('--batch_size', type=int, default=5, help='')
parser.add_argument('--max_epochs', type=int, default=50, help='')
parser.add_argument('--num_workers', type=int, default=1, help='')
parser.add_argument('--loss_type', type=str, default='ssdu', help='')

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


    lr = torch.linspace(1e-5, 0.01, 500)
    losses = []
    sample = next(iter(train_loader))
    mask, undersampled_slice, sampled_slice, ssdu_indecies = to_device(sample, current_device, args.loss_type)
    for lrs in lr:
        optimizer = torch.optim.Adam(model.parameters(), lr=lrs)
        output = model(undersampled_slice, mask)
        loss = loss_fn(torch.view_as_real(output * ssdu_indecies), torch.view_as_real(sampled_slice * ssdu_indecies))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    plt.plot(lr, losses)
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
            to_tensor(),
            normalize()
        )
    )
    train_dataset = SelfSupervisedSampling(
        '/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/multicoil_train',
        transforms=transforms,
        R=4,
        R_hat=2
        )
    
    val_dataset = SelfSupervisedSampling(
        '/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/multicoil_val',
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


def to_device(data: Dict, device: str, loss_type: bool):
    """ moves tensors in data to the device specified

    Args:
        data (Dict): data dictionary returned from dataloader
        device (str): device to move data onto
        supervised (bool): return supervised data if true

    Returns:
        torch.Tensor: returns multiple tensors now on the device
    """

    if loss_type == 'supervised':
        input_slice = data['undersampled']
        target_slice = data['k_space']
        mask = data['mask']
        loss_mask = torch.ones_like(mask)
    else:
        input_slice = data['double_undersample']
        target_slice = data['undersampled']
        mask_lambda = data['omega_mask']
        mask_omega = data['mask']

        if loss_type == 'nosier2noise':
            loss_mask = mask_omega
            mask = mask_omega * mask_labmda
        elif loss_type == 'ssdu' or 'k-weighted':
            loss_mask = (~mask_lambda * mask_omega).detach()
            mask = (mask_lambda * mask_omega)

        if loss_type == 'k-weighted':
            loss_mask = loss_mask.type(torch.float32)
            loss_mask /= torch.sqrt(1 - data['K'])
    
    
    loss_mask = loss_mask.unsqueeze(1).repeat(1, input_slice.shape[1], 1, 1).to(device)
    mask = mask.unsqueeze(1).repeat(1, input_slice.shape[1], 1, 1).to(device)

    input_slice = input_slice.to(device)
    target_slice = target_slice.to(device)
    mask = mask.to(device)
    loss_mask = loss_mask.to(device)

    return mask, input_slice, target_slice, loss_mask





if __name__ == '__main__':
    main()
