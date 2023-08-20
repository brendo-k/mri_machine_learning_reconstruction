import torch
import matplotlib.pyplot as plt

from test_varnet import test
from train_varnet_self_supervised import setup_model_backbone
from ml_recon.models import Unet, ResNet, DnCNN, SwinUNETR, Varnet


supervised_paths = {
'unet_supervised': '',
'dncnn_supervised': '',
'resnet_supervised': '',
'transformer_supervised': '',
}

self_supervised_paths = {
'unet_self': '',
'dncnn_self': '',
'resnet_self': '',
'transformer_self': '',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/'

metric_values = {
    'Supervised': (),
    'Unsupervised': (),
}

for model_name, weight_path in supervised_paths.items():
    backbone = model_name.split('_')[0]
    model = setup_model_backbone(backbone, device)
    nmrse, ssim, psnr = test(weight_path, data_dir, model)

import matplotlib.pyplot as plt
import numpy as np

model_types = ("unet", "dncnn", "resnet", "transformer")

x = np.arange(len(model_types))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in metric_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Length (mm)')
ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, model_types)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 250)

plt.show()
