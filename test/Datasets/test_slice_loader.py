from ml_recon.dataset.undersampled_slice_loader import UndersampledSliceDataset
from ml_recon.dataset.filereader.read_h5 import H5FileReader
from torch.utils.data import DataLoader

import torch
import numpy as np
from ml_recon.transforms import toTensor, pad, pad_recon
from torchvision.transforms import Compose
from ml_recon.utils.collate_function import collate_fn
from ml_recon.utils.read_headers import make_header
import pytest

@pytest.fixture(scope="session")
def build_header(tmp_path_factory) -> str:
    path = tmp_path_factory.getbasetemp()
    header_path = make_header('/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/train/', path / 'header.json')
    return header_path

@pytest.fixture
def build_dataset(build_header) -> UndersampledSliceDataset:
    torch.manual_seed(0)
    dataset = UndersampledSliceDataset(build_header, 4)
    dataset.set_file_reader(H5FileReader)
    return dataset

def test_slice_load(build_dataset):
    data = next(iter(build_dataset))
    assert set(data.keys()) == {'k_space', 'recon', 'mask', 'prob_omega', 'undersampled'}
    assert data['k_space'].ndim == 3
    assert data['recon'].ndim == 2

def test_undersampled_slice(build_dataset):
    data = next(iter(build_dataset))
    # produce mask array with same number of channels
    mask_all_channels = np.tile(data['mask'], (data['undersampled'].shape[0], 1, 1))
    # find locations that equal to zero (masked)
    masked_locations = data['undersampled']*np.invert(mask_all_channels) == 0
    equality = np.full(masked_locations.shape, True, dtype=bool)
    np.testing.assert_array_equal(equality, masked_locations, 'All mask locations should be zero!')

    phase_encode_size = data['mask'].shape[-1]
    center = np.floor(phase_encode_size/2).astype(int)
    acs = data['undersampled'][..., center - 5:center + 5]
    assert np.nonzero(acs)

# Test if we are able to batch slices. This requires some overhead by padding reconstruction and k-space so it is 
# same number of dimensions. Pads the coil dimensions as zeros
def test_undersampled_slice_batching(build_header):
    transforms = Compose((pad((640, 320)), pad_recon((320, 320)), toTensor()))
    dataset = UndersampledSliceDataset(build_header, R=4, transforms=transforms)
    dataset.set_file_reader(H5FileReader)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=5)
    data = next(iter(dataloader))
    assert data['k_space'].shape[0] == 5
    assert data['k_space'].ndim == 4

        
# Test if we are able to batch slices. This requires some overhead by padding reconstruction and k-space so it is 
# same number of dimensions. Pads the coil dimensions as zeros
def test_filter(build_header):
    dataset = UndersampledSliceDataset(build_header, R=4, raw_sample_filter=lambda sample: sample['coils'] >= 16, transforms=toTensor())
    dataset.set_file_reader(H5FileReader)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1)
    data = next(iter(dataloader))
    assert data['k_space'].shape[0] == 1
    assert data['k_space'].ndim == 4

def test_probability_mask(build_dataset):
    dataset = build_dataset
    pdf = dataset.gen_pdf_columns(300, 300, 1/4, 8, 10)
    
    
    np.testing.assert_allclose(np.mean(pdf), 1/4, atol=1e-3)
    np.testing.assert_equal(pdf[:, 150 - 5, 150 + 5], np.ones(300, 10))