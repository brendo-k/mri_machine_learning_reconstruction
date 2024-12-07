import torch
import pytest
from functools import partial

from ml_recon.models.MultiContrastVarNet import MultiContrastVarNet, VarnetBlock, VarnetConfig
from ml_recon.models import Unet


@pytest.fixture
def varnet_model() -> MultiContrastVarNet:
    return MultiContrastVarNet(VarnetConfig(contrast_order=['t1']))

@pytest.fixture
def unet_model() -> Unet:
    return Unet(2, 2, depth=1, chans=1)

@torch.no_grad()
def test_varnet_forward(varnet_model):
    with torch.no_grad():
        reference_k = torch.randn(1, 1, 10, 256, 256, dtype=torch.complex64)  # Example reference k-space input
        mask = torch.rand(1, 1, 10, 256, 256) > 0.5 # Example mask input

        output_k = varnet_model.forward(reference_k, mask)

    assert output_k.shape == reference_k.shape

@torch.no_grad()
def test_varnet_block_forward(unet_model):
    with torch.no_grad():
        unet = unet_model
        varnet_block = VarnetBlock(unet)
        images = torch.randn(1, 1, 4, 256, 256, dtype=torch.complex64)  # Example images input
        sensetivities = torch.randn(1, 1, 4, 256, 256) > 0.5  # Example sensetivities input

        output_images = varnet_block.forward(images, sensetivities)

    assert output_images.shape == images.shape


def test_varnet_backwards(varnet_model):
    # PREPARE
    reference_k = torch.randn(1, 1, 4, 256, 256, dtype=torch.complex64)  # Example reference k-space input
    mask = torch.ones(1, 1, 4, 256, 256, dtype=torch.bool)  # Example mask input
    label = torch.rand(1, 1, 4, 256, 256, 2)

    # ARANGE
    output_k = varnet_model.forward(reference_k, mask)
    output_k = torch.view_as_real(output_k)
    
    optim = torch.optim.Adam(varnet_model.parameters(), lr=1e-3)
    loss = torch.nn.functional.mse_loss(output_k, label)
    loss.backward()
    optim.step()

    output2_k = varnet_model.forward(reference_k, mask)
    loss2 = torch.nn.functional.mse_loss(torch.view_as_real(output2_k), label)

    assert loss2 < loss

@torch.no_grad()
def test_norm(unet_model):
    block = VarnetBlock(unet_model)
    x = torch.randn(5, 4, 128, 128)
    x_norm, mean, std = block.norm(x)
    
    torch.testing.assert_close(x_norm.mean((-1, -2)), torch.zeros(5, 4))
    torch.testing.assert_close(x_norm.std((-1, -2)), torch.ones(5, 4))
    
    x_through = block.unnorm(x_norm, mean, std)

    torch.testing.assert_close(x, x_through)




@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("contrasts", [1, 3])
@pytest.mark.parametrize("height", [32, 64])
def test_varnet_different_sizes(batch_size, contrasts, height):
    """Test VarNet_mc with different input sizes"""
    channels = 4
    width = height
    contrast_order = ['t1' for _ in range(contrasts)]
    model = MultiContrastVarNet(
        VarnetConfig(contrast_order=contrast_order)
    )
    
    k_space = torch.complex(
        torch.randn(batch_size, contrasts, channels, height, width),
        torch.randn(batch_size, contrasts, channels, height, width)
    )
    mask = torch.ones(batch_size, contrasts, channels, height, width)
    
    output = model(k_space, mask)
    
    assert output.shape == (batch_size, contrasts, channels, height, width)
    assert not torch.isnan(output).any()
    
    
def test_lambda_reg_gradient():
    """Test that lambda_reg parameters are properly updated during training"""
    model = MultiContrastVarNet(VarnetConfig(['t1']))
    
    # Store initial lambda_reg values
    initial_lambda = model.lambda_reg.clone()
    
    # Create sample data
    k_space = torch.complex(
        torch.randn(2, 1, 4, 32, 32),
        torch.randn(2, 1, 4, 32, 32)
    )
    mask = torch.ones(2, 1, 4, 32, 32)
    
    # Forward pass and backward pass with dummy loss
    output = model(k_space, mask)
    loss = output.abs().mean()
    loss.backward()
    
    # Check that gradients exist and lambda_reg hasn't changed yet
    assert model.lambda_reg.grad is not None
    assert torch.allclose(model.lambda_reg, initial_lambda)