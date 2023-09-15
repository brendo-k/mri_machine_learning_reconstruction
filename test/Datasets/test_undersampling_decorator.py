import numpy as np
import pytest

import torch
from torch.utils.data import DataLoader

from ml_recon.utils.collate_function import collate_fn
from ml_recon.dataset.sliceloader import SliceDataset
from ml_recon.dataset.filereader.read_h5 import H5FileReader


@pytest.fixture
def build_dataset(get_data_dir, scope='session') -> SliceDataset:
    torch.manual_seed(0)
    dataset = SliceDataset(get_data_dir, build_new_header=True)
    return dataset

@pytest.fixture
def get_data_dir() -> str:
    return '/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/multicoil_train/'

def test_slice_load(build_dataset):
    data = next(iter(build_dataset))
    assert len(data) == 4

def test_undersampled_slice(build_dataset):
    data = next(iter(build_dataset))

    phase_encode_size = data[0].shape[-1]
    center = np.floor(phase_encode_size/2).astype(int)
    acs = data[0][..., center - 5:center + 5]
    assert (acs != 0).all()

# Test if we are able to batch slices. This requires some overhead by padding reconstruction and k-space so it is 
# same number of dimensions. Pads the coil dimensions as zeros
def test_undersampled_slice_batching(get_data_dir):
    dataset = SliceDataset(get_data_dir)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=5)
    data = next(iter(dataloader))
    assert data[0].shape[0] == 5
    assert data[0].ndim == 4


def test_non_deterministic(get_data_dir):
    dataset = SliceDataset(get_data_dir, R=4)

    data1 = dataset[0]
    data2 = dataset[0]

    assert ((data1[1] != 0) != (data2[1] != 0)).any()

def test_non_deterministic_between_slices(get_data_dir):
    dataset = SliceDataset(get_data_dir, R=4)

    data1 = dataset[0]
    data2 = dataset[1]

    assert ((data1[1] != 0) == (data2[1] != 0)).any()

    dataset = SliceDataset(get_data_dir, R=4)

    data1 = dataset[0]
    data2 = dataset[1]
    
    assert ((data1[1] != 0) == (data2[1] != 0)).any()

def test_undersampling(get_data_dir):
    dataset = SliceDataset(get_data_dir, R=4)

    doub_under, under, k_space, k = dataset[0]

    assert (doub_under == 0).sum() > (under == 0).sum()
    assert (under == 0).sum() > (k_space == 0).sum()
    

