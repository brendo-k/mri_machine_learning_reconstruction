import os
import argparse
import matplotlib.pyplot as plt

import torch
from train_model import prepare_data, setup_model_backbone, to_device
from train_utils import setup_devices
from ml_recon.dataset.Brats_dataset import BratsDataset
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator


from torch.distributed import init_process_group, destroy_process_group

def main():
    args = parser.parse_args()
    args.data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset/'
    args.contrasts = ['t1', 't2', 't1ce', 'flair']
    
    current_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, test_loader = prepare_data(args, False)
    model = setup_model_backbone('unet', current_device, input_channels=len(args.contrasts)*2)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1, step_size_up=200, step_size_down=0, cycle_momentum=False)

    losses = []
    lr = []

    for i in range(200):
        print(scheduler.get_last_lr())
        optimizer.zero_grad()
        data = next(iter(train_loader))
        mask, input_slice, target_slice, loss_mask, zf_mask = to_device(data, current_device, args.loss_type)
        
        output = model(input_slice, mask)
        loss = loss_fn(torch.view_as_real(output * loss_mask), torch.view_as_real(target_slice * loss_mask))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        lr.append(scheduler.get_last_lr())
        scheduler.step()

    plt.plot(lr, losses)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('/home/kadotab/python/ml/loss_vs_lr.png')

   


if __name__ == '__main__':

    # Argparse
    parser = argparse.ArgumentParser(description='Varnet self supervised trainer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate to use')
    parser.add_argument('--batch_size', type=int, default=6, help='')
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
    parser.add_argument('--dataset', type=str, default='brats', help='')
    parser = BratsDataset.add_model_specific_args(parser)
    parser = UndersampleDecorator.add_model_specific_args(parser)
    main()
