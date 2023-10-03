import torch
import numpy as np
import pytest
import os

from ml_recon.dataset.Brats_dataset import BratsDataset
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator

@pytest.fixture
def brats_dataset() -> BratsDataset:
    torch.manual_seed(0)
    dataset = BratsDataset('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/subset/train/')
    return dataset

def test_init(brats_dataset):
    data_list = brats_dataset.data_list[0]
    keys = data_list.keys()
    assert 'flair' in keys
    assert 't1' in keys
    assert 't1ce' in keys
    assert 't2' in keys

    dir_names = [os.path.dirname(file) for file in  data_list.values()]
    # should all be the same values
    assert len(set(dir_names)) == 1

def test_apply_sensetivites(brats_dataset):
    x = np.random.rand(5, brats_dataset.nx, brats_dataset.ny)
    x_sense = brats_dataset.apply_sensetivities(x) 

    assert x_sense.dtype == np.complex_
    assert x_sense.ndim == 4
    assert x_sense.shape == (5, 8, brats_dataset.nx, brats_dataset.ny)

def test_generate_phase(brats_dataset):
    x = np.random.rand(5, brats_dataset.nx, brats_dataset.ny) + 1j * np.random.rand(5, brats_dataset.nx, brats_dataset.ny)
    x_phase = brats_dataset.generate_and_apply_phase(x)

def test_samples(brats_dataset):
    dataset = UndersampleDecorator(brats_dataset)
    doub_under, under, k_space, k = dataset[0]
    
    assert doub_under.shape == (4, 8, 256, 256)
    assert under.shape == (4, 8, 256, 256)
    assert k_space.shape == (4, 8, 256, 256)

    assert doub_under.dtype == torch.complex64
    assert under.dtype == torch.complex64
    assert k_space.dtype == torch.complex64

def test_undersampling_all_same(brats_dataset):
    dataset = UndersampleDecorator(brats_dataset)
    doub_under, under, k_space, k = dataset[0]
    

    doub_mask = doub_under == 0
    reference_slice = doub_mask[:, 0, :, :]
    torch.testing.assert_close(doub_mask, reference_slice.unsqueeze(1).expand_as(doub_mask))

    under_mask = under == 0
    reference_slice = under_mask[:, 0, :, :]
    torch.testing.assert_close(under_mask, reference_slice.unsqueeze(1).expand_as(under_mask))

    k_space_mask = k_space == 0
    reference_slice = k_space_mask[:, 0, :, :]
    torch.testing.assert_close(k_space_mask, reference_slice.unsqueeze(1).expand_as(k_space_mask))
