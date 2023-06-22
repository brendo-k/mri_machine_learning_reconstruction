import torch
import pytest

from ml_recon.models.varnet import VarNet, VarnetBlock
from ml_recon.models.NormUnet import NormUnet


@pytest.fixture
def varnet_model():
    return VarNet()

@pytest.fixture(scope="session")
def build_header(tmp_path_factory):
    path = tmp_path_factory.getbasetemp()
    header_path = make_header('/home/kadotab/projects/def-mchiew/kadotab/Datasets/t1_fastMRI/multicoil_train/16_chans/train/', path / 'header.json')
    return header_path

@pytest.fixture
def build_dataset(build_header):
    torch.manual_seed(0)
    dataset = UndersampledSliceDataset(build_header, 4)
    dataset.set_file_reader(H5FileReader)
    return dataset

def test_varnet_forward(varnet_model):
    with torch.no_grad():
        reference_k = torch.randn(1, 2, 256, 256, dtype=torch.complex64)  # Example reference k-space input
        mask = torch.ones(1, 256, 256) > 0.5 # Example mask input

        output_k = varnet_model.forward(reference_k, mask)

    assert output_k.shape == reference_k.shape

def test_varnet_block_forward():
    with torch.no_grad():
        unet = NormUnet(2, 2)
        varnet_block = VarnetBlock(unet)
        images = torch.randn(1, 2, 256, 256, dtype=torch.complex64)  # Example images input
        sensetivities = torch.randn(1, 2, 256, 256) > 0.5  # Example sensetivities input

        output_images = varnet_block.forward(images, sensetivities)

    assert output_images.shape == images.shape


def test_varnet_backwards(varnet_model):
    # PREPARE
    reference_k = torch.randn(1, 1, 256, 256, dtype=torch.complex64)  # Example reference k-space input
    mask = torch.ones(1, 256, 256)  # Example mask input
    label = torch.rand(1, 1, 256, 256, 2)

    # ARANGE
    output_k = varnet_model.forward(reference_k, mask)
    
    optim = torch.optim.Adam(varnet_model.parameters(), lr=1e-3)
    loss = torch.nn.functional.mse_loss(output_k, label)
    loss.backward()
    optim.step()

    output2_k = varnet_model.forward(reference_k, mask)
    loss2 = torch.nn.functional.mse_loss(output2_k, label)

    assert loss2 < loss