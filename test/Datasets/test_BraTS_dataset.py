import torch
import numpy as np
import pytest
import os

from ml_recon.dataset.BraTS_dataset import BratsDataset
from ml_recon.dataset.undersample_decorator import UndersampleDecorator

@pytest.fixture
def brats_dataset() -> BratsDataset:
    path = './test/test_data/simulated_subset_random_phase/train/'
    dataset = BratsDataset(path, nx=128, ny=128)
    return dataset


def test_init(brats_dataset):
    contrast_order = brats_dataset.contrast_order
    assert 'flair' in contrast_order
    assert 't1' in contrast_order
    assert 't1ce' in contrast_order
    assert 't2' in contrast_order


def test_samples(brats_dataset):
    dataset = UndersampleDecorator(brats_dataset)
    data= dataset[0]
    under = data.input
    k_space = data.fs_k_space
    
    assert under.shape == (4, 10, 128, 128)
    assert k_space.shape == (4, 10, 128, 128)

    assert under.dtype == torch.complex64
    assert k_space.dtype == torch.complex64
    

