from ml_recon.Dataset.self_supervised_slice_loader import SelfSupervisedSampling
from ml_recon.Dataset.undersampled_slice_loader import UndersampledSliceDataset
from ml_recon.Dataset.FileReader.read_h5 import H5FileReader
from torch.utils.data import DataLoader
import torch
import numpy as np
from ml_recon.Transforms import toTensor, pad, pad_recon
from torchvision.transforms import Compose
from ml_recon.Utils.collate_function import collate_fn

def test_slice_load():
    dataset = SelfSupervisedSampling('/home/kadotab/header.json', 4, 2)
    dataset.set_file_reader(H5FileReader)
    data = next(iter(dataset))
    assert 'double_undersample' in data.keys()
    assert 'delta_mask' in data.keys()
    assert data['double_undersample'].ndim == 3
    assert data['delta_mask'].ndim == 2
    assert (data['double_undersample'] == 0).sum() > (data['undersampled'] == 0).sum()
