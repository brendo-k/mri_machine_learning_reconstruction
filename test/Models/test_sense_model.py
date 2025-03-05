from ml_recon.models.SensetivityModel_mc import SensetivityModel_mc
import torch
import pytest


@pytest.fixture
def sensitivity_model():
    """Fixture to create a SensetivityModel_mc instance for testing."""
    return SensetivityModel_mc(in_chans=2, out_chans=2, chans=18)

@torch.no_grad()
def test_passthrough(sensitivity_model):
    x = torch.rand((2, 1, 6, 640, 320))
    mask = torch.rand((2, 1, 6, 640, 320))
    output = sensitivity_model(x, mask)

    assert output.shape == x.shape


@torch.no_grad()
def test_scaling():
    x = torch.rand((2, 1, 20, 640, 320))
    mask = torch.zeros(2, 1, 20, 640, 320)
    mask[..., 150:170] = 1
    mask = mask.to(torch.bool)

    sense_model = SensetivityModel_mc(2, 2, 4)
    x_masked = sense_model(x, mask)

    x_summed = (x_masked * x_masked.conj()).sum(2)

    torch.testing.assert_close(x_summed, torch.ones_like(x_summed))


@torch.no_grad()
def test_mask_symmetric_mask(sensitivity_model):
    """
    Test that the mask function creates a symmetric mask around the center.
    
    Checks that:
    1. The mask is symmetric
    2. The number of low frequencies is consistent
    3. The mask is applied correctly
    """
    # Create a sample k-space tensor
    batch_size, num_contrasts, num_channels, height, width = 2, 3, 4, 64, 64
    coil_k_spaces = torch.randn(batch_size, num_contrasts, num_channels, height, width, dtype=torch.complex64)
    
    # Create a center mask
    center_mask = torch.zeros_like(coil_k_spaces, dtype=torch.float)
    center_mask[..., height//2-10:height//2+10, width//2-20:width//2+20] = 1.0
    
    # Apply mask
    masked_k_space = sensitivity_model.mask_center(coil_k_spaces, center_mask)
    
    # Assertions
    assert masked_k_space.shape == coil_k_spaces.shape
    
    # Check symmetry and low-frequency preservation
    for i in range(batch_size):
        for j in range(num_contrasts):
            center_x = width // 2
            center_y = height // 2
            
            # Get the masked region
            masked_region = masked_k_space[i, j, 0, center_y, :]
            
            # Count non-zero elements on left and right sides
            left_nonzero = torch.sum(masked_region[:center_x] != 0)
            right_nonzero = torch.sum(masked_region[center_x:] != 0)
            
            assert left_nonzero == right_nonzero, "Mask is not symmetric"

def test_mask_minimum_low_frequencies(sensitivity_model):
    """
    Test that the mask ensures a minimum number of low frequencies.
    
    Checks that:
    1. Even if the mask suggests fewer low frequencies, at least 10 are preserved
    """
    # Create a k-space tensor with a very narrow mask
    batch_size, num_contrasts, num_channels, height, width = 2, 3, 4, 64, 64
    coil_k_spaces = torch.randn(batch_size, num_contrasts, num_channels, height, width, dtype=torch.complex64)
    
    # Create a very narrow center mask
    center_mask = torch.zeros_like(coil_k_spaces, dtype=torch.float)
    center_mask[..., height//2-1:height//2+1, width//2-1:width//2+1] = 1.0
    
    # Apply mask
    masked_k_space = sensitivity_model.mask_center(coil_k_spaces, center_mask)
    
    # Assertions
    assert masked_k_space.shape == coil_k_spaces.shape
    
    for i in range(batch_size):
        for j in range(num_contrasts):
            center_x = width // 2
            center_y = height // 2
            
            # Get the masked region
            masked_region = masked_k_space[i, j, 0, center_y, :]
            
            # Count non-zero elements
            nonzero_count = torch.sum(masked_region != 0)
            
            assert nonzero_count >= 10, f"Minimum 10 low frequencies not preserved. Actual: {nonzero_count}"

