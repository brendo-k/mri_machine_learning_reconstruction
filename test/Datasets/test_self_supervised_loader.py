import pytest
import torch

from ml_recon.dataset.self_supervised_slice_loader import SelfSupervisedSampling
from ml_recon.dataset.filereader.read_h5 import H5FileReader
from ml_recon.utils.read_headers import make_header

@pytest.fixture
def get_data_dir():
    path = '/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/multicoil_train/'
    return path

@pytest.fixture
def build_dataset(get_data_dir):
    torch.manual_seed(0)
    dataset = SelfSupervisedSampling(get_data_dir, 4, 2)
    return dataset
    

def test_slice_load(build_dataset):
    data = next(iter(build_dataset))
    assert 'double_undersample' in data.keys(), 'add double undersample keys'
    assert 'omega_mask' in data.keys(), 'add lambda mask keys'
    assert data['double_undersample'].ndim == 3, 'double undersample should have (chan, frequency encode, phase encode)'
    assert data['omega_mask'].ndim == 2, 'lambda mask should be 2 dimensional, (frequency encode, phase encode)'
    assert (data['double_undersample'] == 0).sum() > (data['undersampled'] == 0).sum(), 'double undersample should be more zeroed than single undersample'


def test_lambda_mask(build_dataset):
    data = next(iter(build_dataset))
    assert (data['double_undersample'] * ~data['omega_mask'] == 0).all()
    mask = data['omega_mask'] * data['mask']
    assert (torch.from_numpy(data['double_undersample'])[0, :, :] != 0).sum() > mask.sum()*0.80 #90 percent because there are some weird zero padded regions
    torch.testing.assert_allclose(data['double_undersample'] * mask, data['double_undersample']) # should not be affected by mask
    torch.testing.assert_allclose(data['undersampled'] * data['omega_mask'], data['double_undersample']) # should not be affected by mask


def test_acs_lines(build_dataset):
    data = next(iter(build_dataset))
    (ro_lines, pe_lines) = data['double_undersample'][0, :, :].shape
    pe_center = pe_lines//2
    ro_center = ro_lines//2
    assert (data['double_undersample'][:, ro_center-5:ro_center+5, pe_center-5:pe_center+5] != 0).all()