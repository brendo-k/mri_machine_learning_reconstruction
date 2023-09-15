from datetime import datetime
from typing import Dict
import os
import argparse
import time
import yaml
import contextlib
import matplotlib.pyplot as plt

from ml_recon.models.varnet import VarNet
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.tensorboard import SummaryWriter

from ml_recon.transforms import to_tensor, normalize, pad_recon, pad, normalize_mean
from ml_recon.dataset.sliceloader import SliceDataset
from ml_recon.dataset.undersample import Undersampling
from ml_recon.utils import save_model, ifft_2d_img
from ml_recon.utils.collate_function import collate_fn
from train_multi_contrast import prepare_data, setup_model_backbone, to_device

from torchvision.transforms import Compose
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Globals
PROFILE = False

# Argparse
parser = argparse.ArgumentParser(description='Varnet self supervised trainer')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate to use')
parser.add_argument('--batch_size', type=int, default=5, help='')
parser.add_argument('--max_epochs', type=int, default=50, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--data_dir', type=str, default='/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/', help='')
parser.add_argument('--model', type=str, choices=['unet', 'resnet', 'dncnn', 'transformer'], default='unet')
parser.add_argument('--loss_type', type=str, choices=['supervised', 'noiser2noise', 'ssdu', 'k-weighted'], default='supervised')
parser.add_argument('--R_hat', type=float, default=2)

parser.add_argument('--init_method', default='tcp://localhost:18888', type=str, help='')
parser.add_argument('--dist_backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=2, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
parser.add_argument('--use_subset', action='store_true', help='')

def main():
    args = parser.parse_args()
    
    current_device, distributed = setup_devices(args)
    train_loader, val_loader = prepare_data(args, distributed)
    model = setup_model_backbone('unet', current_device)

    if distributed:
        print(f'Setting up DDP in device {current_device}')
        model = DDP(model, device_ids=[current_device])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)


    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    lr = torch.linspace(1e-5, 0.01, 500)
    losses = []
    sample = next(iter(train_loader))
    for lrs in lr:
        data = next(iter(train_loader))
        mask, input_slice, target_slice, loss_mask, zf_mask = to_device(data, current_device, args.loss_type)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lrs)
        output = model(input_slice, mask)
        loss = loss_fn(torch.view_as_real(output * loss_mask), torch.view_as_real(target_slice * loss_mask))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    plt.plot(lr, losses)
    plt.xscale('log')
    plt.yscale('log')
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

if __name__ == '__main__':
    main()
