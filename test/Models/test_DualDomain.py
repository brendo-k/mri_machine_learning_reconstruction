import pytest
import torch
from ml_recon.models.MultiContrastVarNet import VarnetConfig
from ml_recon.models.TriplePathway import TriplePathway, DualDomainConifg

# Fixtures for inputs
@pytest.fixture
def mock_inputs():
    batch_size = 2
    channels = 1
    height = 64
    width = 64
    undersampled_k = torch.rand(batch_size, channels, 2, height, width)
    fully_sampled_k = torch.rand(batch_size, channels, 2, height, width)
    input_set = torch.rand(batch_size, channels, 2, height, width)
    target_set = torch.rand(batch_size, channels, 2, height, width)
    return undersampled_k, fully_sampled_k, input_set, target_set

@pytest.fixture
def dual_domain_config():
    return DualDomainConifg(is_pass_inverse=True, is_pass_original=True)

@pytest.fixture
def mock_varnet_config():
    return VarnetConfig(
        contrast_order=['t1'],
        cascades=3, 
        channels=8
        )  

@pytest.fixture
def dual_domain_ssl(dual_domain_config, mock_varnet_config):
    return TriplePathway(dual_domain_config, mock_varnet_config)

# Test forward method
def test_forward(dual_domain_ssl, mock_inputs):
    undersampled_k, fully_sampled_k, input_set, target_set = mock_inputs

    output = dual_domain_ssl(
        undersampled_k=undersampled_k,
        fully_sampled_k=fully_sampled_k,
        input_set=input_set,
        target_set=target_set,
    )

    assert isinstance(output, dict)
    assert "full_path" in output
    assert "inverse_path" in output
    assert "lambda_path" in output
    if dual_domain_ssl.config.is_pass_inverse:
        assert output["inverse_path"] is not None
    if dual_domain_ssl.config.is_pass_original:
        assert output["full_path"] is not None
    assert output["lambda_path"] is not None

# Test pass_through_model
def test_pass_through_model(dual_domain_ssl, mock_inputs):
    undersampled_k, _, fully_sampled_k, _ = mock_inputs
    mask = torch.ones_like(undersampled_k)

    output = dual_domain_ssl.pass_through_model(undersampled_k, mask, fully_sampled_k)

    assert output.shape == undersampled_k.shape
    assert torch.all(output[fully_sampled_k == 0] == 0)  # Check zero padding applied

# Test final_dc_step
def test_final_dc_step(dual_domain_ssl, mock_inputs):
    undersampled_k, _, _, _ = mock_inputs
    mask = torch.ones_like(undersampled_k)
    estimated = torch.rand_like(undersampled_k)

    output = dual_domain_ssl.final_dc_step(undersampled_k, estimated, mask)

    assert output.shape == estimated.shape
    assert torch.all(output[mask == 1] == undersampled_k[mask == 1])  # Masked areas should match undersampled

# Test create_inverted_masks
def test_create_inverted_masks(dual_domain_ssl, mock_inputs):
    _, _, lambda_set, inverse_set = mock_inputs

    mask_inverse_w_acs, mask_lambda_wo_acs = dual_domain_ssl.create_inverted_masks(lambda_set, inverse_set, 10)

    _, _, _, h, w = lambda_set.shape
    
    slice_y = slice(h//2-5, h//2+5)
    slice_x = slice(w//2-5, w//2+5)
    # Ensure pass through center region works
    torch.testing.assert_close(mask_inverse_w_acs[:, :, :, slice_y, slice_x], lambda_set[:, :, :, slice_y, slice_x])
    torch.testing.assert_close(mask_lambda_wo_acs[:, :, :, slice_y, slice_x], inverse_set[:, :, :, slice_y, slice_x])

    # ensure the remaining values are the same
    torch.testing.assert_close(mask_inverse_w_acs[:, :, :, :h//2-5, :], inverse_set[:, :, :, :h//2-5, :])
    torch.testing.assert_close(mask_lambda_wo_acs[:, :, :, :h//2-5, :], lambda_set[:, :, :, :h//2-5, :])

    torch.testing.assert_close(mask_inverse_w_acs[:, :, :, h//2+5:, :], inverse_set[:, :, :, h//2+5:, :])
    torch.testing.assert_close(mask_lambda_wo_acs[:, :, :, h//2+5:, :], lambda_set[:, :, :, h//2+5:, :])

    torch.testing.assert_close(mask_inverse_w_acs[:, :, :, :, :w//2-5], inverse_set[:, :, :, :, :w//2-5])
    torch.testing.assert_close(mask_lambda_wo_acs[:, :, :, :, :w//2-5], lambda_set[:, :, :, :, :w//2-5])

    torch.testing.assert_close(mask_inverse_w_acs[:, :, :, :, w//2+5:], inverse_set[:, :, :, :, w//2+5:])
    torch.testing.assert_close(mask_lambda_wo_acs[:, :, :, :, w//2+5:], lambda_set[:, :, :, :, w//2+5:])