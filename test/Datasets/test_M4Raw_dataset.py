import pytest
import numpy as np
from ml_recon.dataset.M4Raw_dataset import M4Raw

def test_fill_missing_k_space():
    # Create a sample k-space data with missing values
    k_space = np.random.randn(3, 8, 256, 256)
    k_space[:, 0, 100:150, 100:150] = 0  # Introduce missing values
    k_space[:, :, :10, :10] = 0  # Introduce missing values

    # Call the method
    filled_k_space = M4Raw.fill_missing_k_space(k_space)

    # Check if the missing values are filled
    assert (filled_k_space[:, :, :10, :10] == 0).all()
    assert (filled_k_space[:, :, 100:150, 100:150] != 0).all() # Check if the missing values are filled

if __name__ == '__main__':
    pytest.main()
