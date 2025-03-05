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
        sigmoid_slope_probability=5.0,
        sigmoid_slope_sampling=200,
    )

    # Initialize SSLModel
    return LearnPartitioning(learn_part_config)

@pytest.fixture
def sample_batch() -> torch.Tensor:
    # Generate a sample input batch
    batch_size = 4
    num_contrasts = 1
    num_coils = 8
    height = 128
    width = 128

    mask = torch.ones((batch_size, num_contrasts, 1, height, width)).float()
    return mask
    
def get_center_slice(lambda_mask, acs_size):
    h, w = lambda_mask.shape[-2], lambda_mask.shape[-1]
    # get the bounding indecies for acs lines
    center_h_start = h // 2 - acs_size // 2
    center_h_end = center_h_start + acs_size
    center_w_start = w // 2 - acs_size // 2
    center_w_end = center_w_start + acs_size
    
    # create the slices
    center_h_box = slice(center_h_start, center_h_end)
    center_w_box = slice(center_w_start, center_w_end)
    return center_h_box, center_w_box    


def test_ssl_disjoint_sets(ssl_model: LearnPartitioning, sample_batch: torch.Tensor):
    """
    Test the forward pass of SSLModel.
    """
    mask = sample_batch
    model = ssl_model

    # Forward pass
    lambda_mask, inverse_mask = model(mask)

    # assertions
    assert lambda_mask.shape == inverse_mask.shape, "Shapes of the masks should be the same"
    assert ((lambda_mask.bool() & inverse_mask.bool()) == 0).all(), "Should have no overlapping points"

    # assertions
def test_lambda_mask_has_acs(ssl_model: LearnPartitioning, sample_batch: torch.Tensor):
    """
    Test the forward pass of SSLModel.
    """
    model = ssl_model
    mask = sample_batch

    # Forward pass
    lambda_mask, inverse_mask = model(mask)

    # assertions
    center_h_box, center_w_box = get_center_slice(lambda_mask, ACS_LINES)

    center_region = lambda_mask[..., center_h_box, center_w_box]
    assert (center_region == 1).all(), 'ACS lines should be all 1!'


def test_updating_params(ssl_model: LearnPartitioning, sample_batch: torch.Tensor):
    """
    Test the forward pass of SSLModel.
    """
    model = ssl_model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    cur_weights = model.sampling_weights.clone()

    # Forward pass
    lambda_mask, inverse_mask = model(sample_batch)
    example_loss = lambda_mask.sum()
    optimizer.zero_grad()
    example_loss.backward()
    optimizer.step()

    assert (cur_weights != model.sampling_weights).any()


