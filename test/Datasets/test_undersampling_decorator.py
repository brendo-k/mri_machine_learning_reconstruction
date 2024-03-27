import numpy as np
import pytest

import torch
from torch.utils.data import DataLoader

from ml_recon.dataset.fastMRI_dataset import FastMRIDataset
from ml_recon.dataset.Brats_dataset import BratsDataset
from ml_recon.dataset.self_supervised_decorator import SelfSupervisedDecorator


ACS_LINES = 10
@pytest.fixture
def dataset(get_data_dir, scope='session') -> SelfSupervisedDecorator:
    dataset = BratsDataset(get_data_dir)
    undersample_dataset = SelfSupervisedDecorator(dataset, R=8, acs_lines=ACS_LINES)
    return undersample_dataset

@pytest.fixture
def get_data_dir() -> str:
    return '/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/simulated_subset_random_phase/train/'

def test_slice_load(dataset):
    dataset = BratsDataset('/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/16_chans/multicoil_train/')
    undersample_dataset = SelfSupervisedDecorator(dataset, acs_lines=ACS_LINES)
    data = next(iter(undersample_dataset))
    assert len(data) == 2

def test_fast_mri(dataset):
    data = next(iter(dataset))
    assert len(data) == 2

def test_undersampled_slice(dataset):
    data = next(iter(dataset))

    phase_encode_size = data[0].shape[-1]
    center = np.floor(phase_encode_size/2).astype(int)
    fe_size = data[0].shape[-2]
    fe_center = np.floor(fe_size/2).astype(int)
    acs = data[0][..., fe_center,  center - ACS_LINES//2:center + ACS_LINES//2]
    assert (acs != 0).all()


def test_non_deterministic(dataset):
    data1 = dataset[0]
    data2 = dataset[0]

    assert ((data1[0] != 0) != (data2[0] != 0)).any()

def test_non_deterministic_between_slices(dataset):
    data1 = dataset[0]
    data2 = dataset[1]
    assert ((data1[1] != 0) != (data2[1] != 0)).any()

def test_undersampling(dataset):
    doub_under, under = dataset[0]

    assert not torch.any((doub_under != 0) & (under != 0))
    
