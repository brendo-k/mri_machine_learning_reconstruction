import pytest
import torch
from ml_recon.models.LearnPartitioning import LearnPartitioning, LearnPartitionConfig
from ml_recon.models.MultiContrastVarNet import VarnetConfig

ACS_LINES = 10
@pytest.fixture
def ssl_model():
    # Configure LearnPartitionConfig
    learn_part_config = LearnPartitionConfig(
        image_size=(1, 128, 128),  # 4 samples, 128x128 image size
        inital_R_value=2.0,
        k_center_region=ACS_LINES,
        is_line_constrained=False,
        sigmoid_slope_probability=5.0,
        sigmoid_slope_sampling=200,
    )

    # Initialize SSLModel
    return LearnPartitioning(learn_part_config)

@pytest.fixture
def sample_batch():
    # Generate a sample input batch
    batch_size = 4
    num_contrasts = 1
    num_coils = 8
    height = 128
    width = 128

    input = torch.randn(batch_size, num_contrasts, num_coils, height, width, dtype=torch.complex64)
    mask = torch.ones((batch_size, num_contrasts, 1, height, width)).float()
    return (input, mask)
    

def test_ssl_model_forward(ssl_model, sample_batch):
    """
    Test the forward pass of SSLModel.
    """
    model = ssl_model
    input, mask = sample_batch

    # Forward pass
    lambda_mask, inverse_mask = model(input, mask)

    # assertions
    assert lambda_mask.shape == inverse_mask.shape, "Shapes of the masks should be the same"
    assert ((lambda_mask.bool() & inverse_mask.bool()) == 0).all(), "Should have no overlapping points"

    # assertions
def test_lambda_mask_has_acs(ssl_model, sample_batch):
    """
    Test the forward pass of SSLModel.
    """
    model = ssl_model
    input, mask = sample_batch

    # Forward pass
    lambda_mask, inverse_mask = model(input, mask)

    # assertions
    _, _, _, h, w = lambda_mask.shape
    center_h_start = h // 2 - 5
    center_h_end = h // 2 + 5
    center_w_start = w // 2 - 5
    center_w_end = w // 2 + 5

    center_region = lambda_mask[..., center_h_start:center_h_end, center_w_start:center_w_end]
    assert (center_region == 1).all(), 'ACS lines should be all 1!'