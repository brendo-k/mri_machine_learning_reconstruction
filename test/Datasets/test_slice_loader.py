from ml_recon.Dataset.slice_loader import SliceLoader
from ml_recon.Dataset.undersampled_slice_loader import UndersampledSliceDataset
from ml_recon.Dataset.FileReader.read_h5 import H5FileReader
from torch.utils.data import DataLoader
import torch
import numpy as np
from ml_recon.Transforms import toTensor, pad, pad_recon
from torchvision.transforms import Compose
from ml_recon.Utils.collate_function import collate_fn

def test_slice_load():
    dataset = SliceLoader('/home/kadotab/header.json')
    dataset.set_file_reader(H5FileReader)
    data = next(iter(dataset))
    assert list(data.keys()) == ['k_space', 'recon']
    assert data['k_space'].ndim == 3
    assert data['recon'].ndim == 2

def test_undersampled_slice():
    dataset = UndersampledSliceDataset('/home/kadotab/header.json', R=4, acs_width=20)
    dataset.set_file_reader(H5FileReader)
    data = next(iter(dataset))
    assert list(data.keys()) == ['k_space', 'recon', 'mask', 'undersampled']
    assert data['k_space'].ndim == 3
    assert data['recon'].ndim == 2
    assert data['mask'].ndim == 2
    # produce mask array with same number of channels
    mask_all_channels = np.tile(data['mask'], (data['undersampled'].shape[0], 1, 1))
    # find locations that equal to zero (masked)
    masked_locations = data['undersampled']*np.invert(mask_all_channels) == 0
    equality = np.full(masked_locations.shape, True, dtype=bool)
    np.testing.assert_array_equal(equality, masked_locations, 'All mask locations should be zero!')

    phase_encode_size = data['mask'].shape[-1]
    center = np.floor(phase_encode_size/2).astype(int)
    acs_bounds = [center - 10, center + 10]
    acs = data['undersampled'][..., acs_bounds[0]:acs_bounds[1]]
    assert acs.shape[-1] == 20
    assert np.nonzero(acs)

# Test if we are able to batch slices. This requires some overhead by padding reconstruction and k-space so it is 
# same number of dimensions. Pads the coil dimensions as zeros
def test_undersampled_slice_batching():
    transforms = Compose((pad((640,320)), pad_recon((320,320)), toTensor()))
    dataset = UndersampledSliceDataset('/home/kadotab/header.json', R=4, transforms=transforms)
    dataset.set_file_reader(H5FileReader)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=5)
    data = next(iter(dataloader))
    assert data['k_space'].shape[0] == 5
    assert data['k_space'].ndim == 4

        
# Test if we are able to batch slices. This requires some overhead by padding reconstruction and k-space so it is 
# same number of dimensions. Pads the coil dimensions as zeros
def test_filter():
    dataset = UndersampledSliceDataset('/home/kadotab/header.json', R=4, raw_sample_filter=lambda sample: sample['coils'] >= 16)
    dataset.set_file_reader(H5FileReader)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1)
    data = next(iter(dataloader))
    assert data['k_space'].shape[0] == 5
    assert data['k_space'].ndim == 4