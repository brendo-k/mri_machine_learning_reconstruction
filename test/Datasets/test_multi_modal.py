import torch
import numpy as np
import pytest
import os

from ml_recon.dataset.Brats_dataset import BratsDataset
from ml_recon.dataset.self_supervised_decorator import SelfSupervisedDecorator

@pytest.fixture
def brats_dataset(request) -> BratsDataset:
    torch.manual_seed(0)
    test_case_params = request.param
    data_dir = test_case_params.get("directory")
    dataset = BratsDataset(data_dir)
    return dataset

test_cases = [
    {"name": "Image Dataset", "directory": "/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/subset/train/"},
    {"name": "Kspace Dataset", "directory": "/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/unchunked/train/"}
]


@pytest.mark.parametrize("brats_dataset", test_cases, indirect=True)
def test_init(brats_dataset):
    contrast_order = brats_dataset.contrast_order
    assert 'flair' in contrast_order
    assert 't1' in contrast_order
    assert 't1ce' in contrast_order
    assert 't2' in contrast_order


@pytest.mark.parametrize("brats_dataset", test_cases, indirect=True)
def test_samples(brats_dataset):
    dataset = SelfSupervisedDecorator(brats_dataset)
    doub_under, under, k_space, k = dataset[0]
    
    assert doub_under.shape == (4, 8, 256, 256)
    assert under.shape == (4, 8, 256, 256)
    assert k_space.shape == (4, 8, 256, 256)

    assert doub_under.dtype == torch.complex64
    assert under.dtype == torch.complex64
    assert k_space.dtype == torch.complex64
    
    first_contrast_mask = doub_under[0, ...] != 0
    for i in range(1, doub_under.shape[0]):
        i_mask = doub_under[i, ...] != 0
        assert not (first_contrast_mask == i_mask).all()

    first_contrast_mask = under[0, ...] != 0
    for i in range(1, doub_under.shape[0]):
        i_mask = under[i, ...] != 0
        assert not (first_contrast_mask == i_mask).all()

@pytest.mark.parametrize("brats_dataset", test_cases, indirect=True)
def test_undersampling_all_same(brats_dataset):
    dataset = SelfSupervisedDecorator(brats_dataset)
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
