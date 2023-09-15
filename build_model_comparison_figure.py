import torch
import matplotlib.pyplot as plt
import os
import numpy as np

from test_varnet import test
from train_multi_contrast import setup_model_backbone
from ml_recon.models import Unet, ResNet, DnCNN, SwinUNETR, VarNet

def main():
    supervised_paths = {
    'unet_supervised': '',
    'dncnn_supervised': '',
    'resnet_supervised': '',
    'transformer_supervised': '',
    }

    self_supervised_paths = {
    'unet_self': '0820-09:11:16VarNet-unet-k-weighted/20230820-120612VarNet.pt',
    'dncnn_self': '0820-09:17:43VarNet-dncnn-k-weighted/20230820-115301VarNet.pt',
    'resnet_self': '0820-09:13:52VarNet-resnet-k-weighted/20230820-124531VarNet.pt',
    #'transformer_self': '0820-09:10:08VarNet-transformer-k-weighted',
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/'

    metric_values_mse = {
        'Supervised': [],
        'Unsupervised': [],
    }

    metric_values_ssim = {
        'Supervised': [],
        'Unsupervised': [],
    }

    metric_values_psnr = {
        'Supervised': [],
        'Unsupervised': [],
    }

    for model_name, weight_path in self_supervised_paths.items():
        backbone = model_name.split('_')[0]
        model = setup_model_backbone(backbone, device)
        weight_path = os.path.join('/home/kadotab/scratch/runs/', weight_path)
        nmrse, ssim, psnr = test(weight_path, data_dir, model)
        print(model_name)
        metric_values_mse['Unsupervised'].append(nmrse.cpu())
        metric_values_ssim['Unsupervised'].append(ssim.cpu())
        metric_values_psnr['Unsupervised'].append(psnr.cpu())
        metric_values_mse['Supervised'].append(0)
        metric_values_ssim['Supervised'].append(0)
        metric_values_psnr['Supervised'].append(0)


    model_types = ("unet", "dncnn", "resnet")

    x = np.arange(3)  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    plt.figure(0)
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in metric_values_mse.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('')
    ax.set_ylim([0.0, 0.05])
    ax.set_title('NMSE of supervised models')
    ax.set_xticks(x + width, model_types)
    ax.legend(loc='upper left', ncols=3)
    plt.savefig('mse')

    plt.figure(1)
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in metric_values_ssim.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Model type')
    ax.set_title('SSIM of supervised models')
    ax.set_xticks(x + width, model_types)
    ax.set_ylim([0, 2])
    ax.legend(loc='upper left', ncols=3)
    plt.savefig('ssim')


    plt.figure(2)
    fig, ax = plt.subplots(layout='constrained')
    multiplier = 0
    for attribute, measurement in metric_values_psnr.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('model type')
    ax.set_title('Psnr of supervised models')
    ax.set_ylim([10, 50])
    ax.set_xticks(x + width, model_types)
    ax.legend(loc='upper left', ncols=3)

    plt.savefig('psnr')



if __name__ == "__main__":
    main()