import pytest
import torch

from ml_recon.dataset.self_supervised_slice_loader import SelfSupervisedSampling
from ml_recon.dataset.filereader.read_h5 import H5FileReader
from ml_recon.utils.read_headers import make_header

@pytest.fixture(scope="session")
def build_header(tmp_path_factory):
    path = tmp_path_factory.getbasetemp()
    header_path = make_header('/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/train/', path / 'header.json')
    return header_path

@pytest.fixture
def build_dataset(build_header):
    torch.manual_seed(0)
    dataset = SelfSupervisedSampling(build_header, 4, 2)
    dataset.set_file_reader(H5FileReader)
    return dataset
    

def test_slice_load(build_dataset):
    data = next(iter(build_dataset))
    assert 'double_undersample' in data.keys(), 'add double undersample keys'
    assert 'lambda_mask' in data.keys(), 'add lambda mask keys'
    assert data['double_undersample'].ndim == 3, 'double undersample should have (chan, frequency encode, phase encode)'
    assert data['lambda_mask'].ndim == 2, 'lambda mask should be 2 dimensional, (frequency encode, phase encode)'
    assert (data['double_undersample'] == 0).sum() > (data['undersampled'] == 0).sum(), 'double undersample should be more zeroed than single undersample'


def test_lambda_mask(build_dataset):
    data = next(iter(build_dataset))
    assert (data['double_undersample'] * ~data['lambda_mask'] == 0).all()
    mask = data['lambda_mask'] * data['mask']
    assert (torch.from_numpy(data['double_undersample'])[0, :, :] != 0).sum() > mask.sum()*0.80 #90 percent because there are some weird zero padded regions
    torch.testing.assert_allclose(data['double_undersample'] * mask, data['double_undersample']) # should not be affected by mask

def test_acs_lines(build_dataset):
    data = next(iter(build_dataset))
    (ro_lines, pe_lines) = data['double_undersample'][0, :, :].shape
    pe_center = pe_lines//2
    ro_center = ro_lines//2
    assert (data['double_undersample'][:, ro_center-5:ro_center+5, pe_center-5:pe_center+5] != 0).all()