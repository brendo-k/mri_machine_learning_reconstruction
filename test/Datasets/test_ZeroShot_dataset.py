import pytest
import numpy as np
import torch
from ml_recon.dataset.Zeroshot_datset import ZeroShotDataset
import h5py
import tempfile
import os

@pytest.fixture
def temp_h5_file():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
        temp_file_path = temp_file.name

    # Open the file in write mode and create the dataset
    with h5py.File(temp_file_path, 'w') as f:
        # dimension of batch channel height widht
        real = np.random.rand(4, 10, 128, 200, dtype=np.float32)
        imag = np.random.rand(4, 10, 128, 200, dtype=np.float32)
        f.create_dataset('kspace', data= real + 1j * imag)

    # Yield the file path for use in tests
    yield temp_file_path

    # Clean up the file after the test
    os.remove(temp_file_path)



def test_sets(temp_h5_file):
    dataset_val = ZeroShotDataset(temp_h5_file, is_validation=True)
    dataset_train = ZeroShotDataset(temp_h5_file, is_validation=False)
    dataset_test = ZeroShotDataset(temp_h5_file, is_validation=False, is_test=True)


    input = dataset_val[0].input
    target = dataset_val[0].target
    ssl_data = dataset_train[0]
    input_train = ssl_data.input
    target_train = ssl_data.target

    under = dataset_test[0].input

    assert not ((input != 0) & (target != 0)).any()
    torch.testing.assert_close(input_train + target_train, input)
    torch.testing.assert_close(under, input + target)

def test_determinism(temp_h5_file): 
    dataset_val = ZeroShotDataset(temp_h5_file, is_validation=True)

    input1 = dataset_val[0].input
    input2 = dataset_val[0].input
    input3 = dataset_val[0].input
    target1 = dataset_val[0].target
    target2 = dataset_val[0].target
    target3 = dataset_val[0].target

    torch.testing.assert_close(input1, input2)
    torch.testing.assert_close(input2, input3)

    torch.testing.assert_close(target1, target2)
    torch.testing.assert_close(target2, target3)

def test_new_masks(temp_h5_file):
    dataset_train = ZeroShotDataset(temp_h5_file, is_validation=False)
    data1 = dataset_train[0]
    data2 = dataset_train[0]
    data3 = dataset_train[0]
    input_train1 = data1.input
    input_train2 = data2.input
    input_train3 = data3.input
    target_train1 = data1.target
    target_train2 = data2.target
    target_train3 = data3.target    
    assert not torch.equal(input_train1, input_train2)
    assert not torch.equal(input_train2, input_train3)

    torch.testing.assert_close(input_train1 + target_train1, input_train2 + target_train2)
    torch.testing.assert_close(input_train2 + target_train2, input_train3 + target_train3)

    

def test_splitting_mask(temp_h5_file):
    dataset_train = ZeroShotDataset(temp_h5_file, is_validation=False)
    mask = np.random.rand(1, 10, 320, 320) < 0.5
    val_mask, train_mask = dataset_train.gen_validation_set(0, mask)

    torch.testing.assert_close(val_mask + train_mask, mask)

    assert val_mask.sum() < mask.sum() * 0.3
    assert val_mask.sum() > mask.sum() * 0.1

def test_same_size(temp_h5_file):
    dataset_train = ZeroShotDataset(temp_h5_file, is_validation=False)
    dataset_val = ZeroShotDataset(temp_h5_file, is_validation=True)
    dataset_test = ZeroShotDataset(temp_h5_file, is_validation=False, is_test=True)

    data_train = dataset_train[0]
    data_val = dataset_val[0]
    data_test = dataset_test[0]

    assert data_train.input.shape == data_train.target.shape
    assert data_train.input.shape == data_train.fs_k_space.shape

    assert data_val.input.shape == data_val.target.shape
    assert data_val.input.shape == data_val.fs_k_space.shape

    assert data_test.input.shape == data_test.target.shape
    assert data_test.input.shape == data_test.fs_k_space.shape

    assert data_test.input.shape == data_val.fs_k_space.shape
    assert data_test.input.shape == data_train.fs_k_space.shape
