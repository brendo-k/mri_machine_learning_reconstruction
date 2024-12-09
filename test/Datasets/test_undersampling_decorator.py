import numpy as np
import pytest

import torch
from torch.utils.data import DataLoader

from ml_recon.dataset.fastMRI_dataset import FastMRIDataset
from ml_recon.dataset.BraTS_dataset import BratsDataset
from ml_recon.dataset.undersample_decorator import UndersampleDecorator


ACS_LINES = 10
@pytest.fixture
def supervised_dataset(get_data_dir) -> UndersampleDecorator:
    dataset = BratsDataset(get_data_dir, nx=128, ny=128)
    undersample_dataset = UndersampleDecorator(dataset, R=8, acs_lines=ACS_LINES, self_supervised=False)
    return undersample_dataset

@pytest.fixture
def get_data_dir() -> str:
    return './test/test_data/simulated_subset_random_phase/train/'

def test_undersampled_slice(supervised_dataset):
    data = next(iter(supervised_dataset))

    width = data.input.shape[-1]
    center_width = np.floor(width/2).astype(int)
    height = data.input.shape[-2]
    center_height = np.floor(height/2).astype(int)

    acs = data.input[..., center_height - ACS_LINES//2:center_height + ACS_LINES + 2,  center_width - ACS_LINES//2:center_width + ACS_LINES//2]
    assert (acs != 0).all()

def test_fully_sampled(supervised_dataset):
    data = supervised_dataset[0]

    assert (data.fs_k_space != 0).all()


def test_non_deterministic(supervised_dataset):
    data1 = supervised_dataset[0]
    data2 = supervised_dataset[0]

    assert ((data1.input != 0) == (data2.input != 0)).any()

def test_non_deterministic_between_slices(supervised_dataset):
    data1 = supervised_dataset[0]
    data2 = supervised_dataset[1]
    assert ((data1.input != 0) != (data2.input != 0)).any()

def test_non_deterministic_between_lambda(get_data_dir):
    dataset = BratsDataset(get_data_dir, nx=128, ny=128)
    undersample_dataset = UndersampleDecorator(dataset, R=4, acs_lines=ACS_LINES, self_supervised=True, R_hat=2)
    
    data1 = undersample_dataset[0]
    data2 = undersample_dataset[0]

    assert ((data1.input != 0) != (data2.input != 0)).any()
    assert ((data1.target != 0) != (data2.target != 0)).any()
    assert ((data1.loss_mask) != (data2.loss_mask)).any()
    assert ((data1.mask) != (data2.mask)).any()
    torch.testing.assert_close(data1.fs_k_space, data2.fs_k_space)

def test_pi_sampling(get_data_dir):
    dataset = BratsDataset(get_data_dir, nx=128, ny=128)
    undersample_dataset = UndersampleDecorator(dataset, R=4, acs_lines=ACS_LINES, self_supervised=False, initial_sampling_method=False)
    
    data1 = undersample_dataset[0]

    mask = data1.input != 0 
    torch.testing.assert_close(mask, data1.mask)
    assert torch.all(mask == mask[...,0,:].unsqueeze(-2))

