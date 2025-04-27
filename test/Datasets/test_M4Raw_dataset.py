import pytest
import numpy as np
from ml_recon.dataset.M4Raw_dataset import M4Raw
import tempfile
import os
import h5py 


SLICES = 10
VOLUMES = 3

@pytest.fixture(scope='session')
def mock_m4raw_dataset_dir():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create subdirectories for each sample
        for i in range(VOLUMES):  # Create sample directories
            file_path = os.path.join(temp_dir, f'data_{i}.h5')
            with h5py.File(file_path, 'w') as f:
                # Create a mock k-space dataset
                kspace_data = np.random.rand(3, SLICES, 4, 256, 256) + 1j * np.random.rand(3, SLICES, 4, 256, 256)
                f.create_dataset('kspace', data=kspace_data)
                
                # Create a mock contrasts dataset
                contrasts = np.array(['t1', 't2', 'flair'], dtype='S')
                f.create_dataset('contrasts', data=contrasts)
        
        yield temp_dir

def test_dataset_length(mock_m4raw_dataset_dir):
    dataloader = M4Raw(mock_m4raw_dataset_dir, contrasts=['t1', 't2', 'flair'])

    assert len(dataloader) == VOLUMES * SLICES 

def test_dataset_outupt(mock_m4raw_dataset_dir):
    dataloader = M4Raw(mock_m4raw_dataset_dir, contrasts=['t1', 't2', 'flair'])
    
    assert isinstance(dataloader[0], np.ndarray)
    assert dataloader[0].shape == (3, 4, 256, 256)
    assert dataloader[0].dtype == np.complex128
    assert (dataloader.contrast_order == ['t1', 't2', 'flair']).all()

def test_single_contrast(mock_m4raw_dataset_dir):
    dataloader = M4Raw(mock_m4raw_dataset_dir, contrasts=['t1'])
    
    assert isinstance(dataloader[0], np.ndarray)
    assert dataloader[0].shape == (1, 4, 256, 256)
    assert dataloader[0].dtype == np.complex128
    assert (dataloader.contrast_order == ['t1']).all()

def test_volumes_not_same(mock_m4raw_dataset_dir):
    dataloader = M4Raw(mock_m4raw_dataset_dir, contrasts=['t1'])
    
    assert (dataloader[0] != dataloader[1]).any(), \
    "should have different values between slices"

if __name__ == '__main__':
    pytest.main()
