import pytest
import numpy as np
from ml_recon.dataset.M4Raw_dataset import M4Raw
import tempfile
import os
import h5py 



@pytest.fixture(scope='session')
def mock_m4raw_dataset_dir():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create subdirectories for each sample
        for i in range(3):  # Create 3 sample directories
            file_path = os.path.join(temp_dir, f'data_{i}.h5')
            with h5py.File(file_path, 'w') as f:
                # Create a mock k-space dataset
                kspace_data = np.random.rand(3, 10, 4, 256, 256) + 1j * np.random.rand(3, 10, 4, 256, 256)
                f.create_dataset('kspace', data=kspace_data)
                
                # Create a mock contrasts dataset
                contrasts = np.array(['t1', 't2', 'flair'], dtype='S')
                f.create_dataset('contrasts', data=contrasts)
        
        yield temp_dir

def test_dataset_outupt(mock_m4raw_dataset_dir):
    dataloader = M4Raw(mock_m4raw_dataset_dir, contrasts=['t1', 't2', 'flair'])
    
    assert len(dataloader) == 30 

    assert dataloader[0].shape == (3, 4, 256, 256)
    assert 't1' in dataloader.contrast_order
    assert 't2' in dataloader.contrast_order
    assert 'flair' in dataloader.contrast_order

if __name__ == '__main__':
    pytest.main()
