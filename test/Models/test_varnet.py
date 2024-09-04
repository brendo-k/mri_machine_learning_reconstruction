import torch
import pytest
from functools import partial

from ml_recon.models.varnet_mc import VarNet_mc, VarnetBlock
from ml_recon.models.unet import Unet


@pytest.fixture
def varnet_model() -> VarNet_mc:
    return VarNet_mc(partial(Unet, 2, 2))

@pytest.fixture
def unet_model() -> Unet:
    return Unet(2, 2, depth=1, chans=1)

def test_varnet_forward(varnet_model):
    with torch.no_grad():
        reference_k = torch.randn(1, 2, 256, 256, dtype=torch.complex64)  # Example reference k-space input
        mask = torch.ones(1, 2, 256, 256) > 0.5 # Example mask input

        output_k = varnet_model.forward(reference_k, mask)

    assert output_k.shape == reference_k.shape

def test_varnet_block_forward(unet_model):
    with torch.no_grad():
        unet = unet_model
        varnet_block = VarnetBlock(unet)
        images = torch.randn(1, 2, 256, 256, dtype=torch.complex64)  # Example images input
        sensetivities = torch.randn(1, 2, 256, 256) > 0.5  # Example sensetivities input

        output_images = varnet_block.forward(images, sensetivities)

    assert output_images.shape == images.shape


#def test_varnet_backwards(varnet_model):
#    # PREPARE
#    reference_k = torch.randn(1, 16, 256, 256, dtype=torch.complex64)  # Example reference k-space input
#    mask = torch.ones(1, 16, 256, 256, dtype=torch.bool)  # Example mask input
#    label = torch.rand(1, 16, 256, 256, 2)
#
#    # ARANGE
#    output_k = varnet_model.forward(reference_k, mask)
#    output_k = torch.view_as_real(output_k)
#    
#    optim = torch.optim.Adam(varnet_model.parameters(), lr=1e-3)
#    loss = torch.nn.functional.mse_loss(output_k, label)
#    loss.backward()
#    optim.step()
#
#    output2_k = varnet_model.forward(reference_k, mask)
#    loss2 = torch.nn.functional.mse_loss(output2_k, label)
#
#    assert loss2 < loss

def test_norm(unet_model):
    block = VarnetBlock(unet_model)
    x = torch.randn(5, 2, 128, 128)
    x_norm, mean, std = block.norm(x)
    x_through = block.unnorm(x_norm, mean, std)

    torch.testing.assert_close(x, x_through)

def test_dc(varnet_model: VarNet_mc):
    x = torch.randn(5, 16, 128, 128)
    mask = torch.zeros_like(x)
    mask[:, :, :, 60:68] = 1
    mask = mask.type(torch.bool)
    dc = varnet_model.data_consistency(x, x, mask)

    torch.testing.assert_close(dc, torch.zeros_like(dc))

def test_dc_subtract(varnet_model: VarNet_mc):
    x = torch.randn(5, 16, 128, 128)
    y = torch.randn(5, 16, 128, 128)
    mask = torch.zeros_like(x)
    mask[:, :, :, 60:68] = 1
    mask = mask.type(torch.bool)
    dc = varnet_model.data_consistency(y, x, mask)
    dc = y - dc
    gt_result = x * mask + y * ~ mask

    torch.testing.assert_close(dc, gt_result)

    
