import torch
import tempfile
import h5py
import numpy as np
import pytest
import os

from ml_recon.dataset.BraTS_dataset import BratsDataset
from ml_recon.dataset.undersample_decorator import UndersampleDecorator

@pytest.fixture
def brats_dataset(mock_brats_dataset_dir) -> BratsDataset:
    path = mock_brats_dataset_dir
    dataset = BratsDataset(path, nx=128, ny=128)
    return dataset

@pytest.fixture
def mock_brats_dataset_dir():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create subdirectories for each sample
        for i in range(3):  # Create 3 sample directories
            sample_dir = os.path.join(temp_dir, f'sample_{i}')
            os.makedirs(sample_dir)
            
            # Create a mock HDF5 file in each sample directory
            file_path = os.path.join(sample_dir, 'data.h5')
            with h5py.File(file_path, 'w') as f:
                # Create a mock k-space dataset
                kspace_data = np.random.rand(10, 4, 256, 256) + 1j * np.random.rand(10, 4, 256, 256)
                f.create_dataset('k_space', data=kspace_data)
                
                # Create a mock contrasts dataset
                contrasts = np.array(['t1', 't2', 'flair', 't1ce'], dtype='S')
                f.create_dataset('contrasts', data=contrasts)
        
        yield temp_dir


def test_init(brats_dataset):
    contrast_order = brats_dataset.contrast_order
    assert 'flair' in contrast_order
    assert 't1' in contrast_order
    assert 't1ce' in contrast_order
    assert 't2' in contrast_order

