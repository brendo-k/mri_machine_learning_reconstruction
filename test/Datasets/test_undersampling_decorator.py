import numpy as np
import pytest

import torch
from torch.utils.data import DataLoader

from ml_recon.utils.collate_function import collate_fn
from ml_recon.dataset.fastMRI_dataset import FastMRIDataset
from ml_recon.dataset.self_supervised_decorator import UndersampleDecorator


ACS_LINES = 10
@pytest.fixture
def dataset(get_data_dir, scope='session') -> UndersampleDecorator:
    torch.manual_seed(0)
    dataset = FastMRIDataset(get_data_dir, build_new_header=True)
    undersample_dataset = UndersampleDecorator(dataset, acs_lines=ACS_LINES)
    return undersample_dataset

@pytest.fixture
def get_data_dir() -> str:
    return '/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/16_chans/multicoil_train/'

def test_slice_load(dataset):
    data = next(iter(dataset))
    assert len(data) == 4

def test_undersampled_slice(dataset):
    data = next(iter(dataset))

    phase_encode_size = data[0].shape[-1]
    center = np.floor(phase_encode_size/2).astype(int)
    acs = data[0][..., center - ACS_LINES//2:center + ACS_LINES//2]
    assert (acs != 0).all()
    assert acs.shape[-1] == 10 

# Test if we are able to batch slices. 
def test_undersampled_slice_batching(dataset):
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=5)
    data = next(iter(dataloader))
    assert data[0].shape[0] == 5
    assert data[0].ndim == 5


def test_non_deterministic(dataset):
    data1 = dataset[0]
    data2 = dataset[0]

    assert ((data1[0] != 0) != (data2[0] != 0)).any()

def test_non_deterministic_between_slices(dataset):
    data1 = dataset[0]
    data2 = dataset[1]

    assert ((data1[1] != 0) != (data2[1] != 0)).any()

def test_undersampling(dataset):
    doub_under, under, k_space, _ = dataset[0]

    assert (doub_under == 0).sum() > (under == 0).sum()
    assert (under == 0).sum() > (k_space == 0).sum()
    
def test_columnwise(dataset):
    doub_under, under, k_space, _ = dataset[0]

    first_k_line_mask = doub_under[0, 0, 0, :] != 0
    torch.testing.assert_close(doub_under != 0, first_k_line_mask.expand_as(doub_under))

    first_k_line_mask = under[0, 0, 0, :] != 0
    torch.testing.assert_close(under != 0, first_k_line_mask.expand_as(doub_under))
    

