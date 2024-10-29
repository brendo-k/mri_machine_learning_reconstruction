import pytest 
import torch 
from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning
from itertools import repeat
import matplotlib.pyplot as plt

IMAGE_SIZE = (2, 128, 128)

@pytest.fixture()
def build_model() -> LearnedSSLLightning:
    return LearnedSSLLightning(IMAGE_SIZE, learned_R=2, contrast_order=['t1', 't2'])


@torch.no_grad()
def test_norm_prob(build_model: LearnedSSLLightning):
    model = build_model
    probability = [torch.rand(IMAGE_SIZE[1:]) for _ in range(IMAGE_SIZE[0])]
    normed_prob = model.norm_2d_probability(
        probability, 
        model.R_value, 
        mask_center=True, 
        center_region=10, 
        image_shape=probability[0].shape
        )
    normed_prob = torch.stack(normed_prob, dim=0)

    torch.testing.assert_close(normed_prob.mean((-1, -2)), torch.tensor([1/2, 1/2]))
    assert normed_prob.max() <= 1 and normed_prob.min() >= 0 

@torch.no_grad()
def test_norm_prob_1d(build_model: LearnedSSLLightning):
    model = build_model
    probability = [torch.rand(IMAGE_SIZE[1]) for _ in range(IMAGE_SIZE[0])]
    normed_prob = model.norm_1d_probability(
        probability, 
        model.R_value, 
        mask_center=True,
        center_region=10, 
        image_shape=IMAGE_SIZE[1:],
        )
    normed_prob = torch.stack(normed_prob, dim=0)

    torch.testing.assert_close(normed_prob.mean((-1)), torch.tensor([1/2, 1/2]))
    assert normed_prob.max() <= 1 and normed_prob.min() >= 0    

@torch.no_grad()
def test_norm_2d_probability_center_masking(build_model):
    # Parameters
    probability = [torch.rand(IMAGE_SIZE[1:]) for _ in range(IMAGE_SIZE[0])]
    center_region = 10
    mask_center = True
    image_shape = IMAGE_SIZE
    
    # Run the method
    normed = build_model.norm_2d_probability(probability, build_model.R_value, center_region, mask_center, image_shape[1:])
    normed = torch.stack(normed, dim=0)
    
    # Verify center region is masked correctly
    center = [dim // 2 for dim in image_shape[1:]]
    center_bb_x = slice(center[0] - center_region // 2, center[0] + center_region // 2)
    center_bb_y = slice(center[1] - center_region // 2, center[1] + center_region // 2)
    assert torch.all(normed[:, center_bb_y, center_bb_x] == 1), "Center region should be masked to 1"

@torch.no_grad()
def test_get_mask(build_model: LearnedSSLLightning):
    model = build_model
    batch_size = 2
    sampling_weights = torch.zeros(IMAGE_SIZE)
    model.sampling_weights = torch.nn.Parameter(sampling_weights)
    mask =  model.get_mask(batch_size, mask_center=True)
    mask_shape = mask.shape

    assert mask_shape == (batch_size, IMAGE_SIZE[0], 1) + IMAGE_SIZE[1:]
    torch.testing.assert_close(mask.mean((-1, -2)), torch.full(mask_shape[:3], 0.5), atol=1e-2, rtol=0)

def test_splitting_sets(build_model: LearnedSSLLightning): 
    model = build_model
    batch_size = 4
    inital_mask = torch.zeros((batch_size, IMAGE_SIZE[0], ) +  (IMAGE_SIZE))
    inital_mask[:, :, ::2, ::2] = 1
    lambda_set, inverse_set = model.split_into_lambda_loss_sets(inital_mask, batch_size)
    lambda_set2, inverse_set2 = model.split_into_lambda_loss_sets(inital_mask, batch_size)

    assert torch.all(((lambda_set == 1) ^ (inverse_set == 1)) == inital_mask)
    assert torch.any(lambda_set != lambda_set2, dim=(-1, -2)).any()
    assert torch.any(inverse_set != inverse_set2, dim=(-1, -2)).any()
