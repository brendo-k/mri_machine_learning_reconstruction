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
    normed_prob = model.norm_prob(probability, model.R_value, mask_center=True)
    normed_prob = torch.stack(normed_prob, dim=0)

    torch.testing.assert_close(normed_prob.mean((-1, -2)), torch.tensor([1/2, 1/2]))
    
    assert normed_prob.max() <= 1 and normed_prob.min() >= 0 

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
    assert torch.all(((lambda_set == 1) | (inverse_set == 1)) == inital_mask)
