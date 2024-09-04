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
    probability = [torch.randn(IMAGE_SIZE[1:]) for _ in range(IMAGE_SIZE[0])]
    normed_prob = model.norm_prob(probability, model.R_value, mask_center=True)
    normed_prob = torch.stack(normed_prob, dim=0)

    torch.testing.assert_close(normed_prob.mean((-1, -2)), torch.tensor([1/2, 1/2]))

@torch.no_grad()
def test_get_mask(build_model: LearnedSSLLightning):
    model = build_model
    batch_size = 2
    sampling_weights = torch.zeros(IMAGE_SIZE)
    mask =  model.get_mask(sampling_weights, batch_size, mask_center=True)
    mask_shape = mask.shape

    assert mask_shape == (batch_size, ) + IMAGE_SIZE
    torch.testing.assert_close(mask.mean((-1, -2)), torch.full(mask_shape[:2], 0.5), atol=1e-2, rtol=0)
    plt.imshow(mask[0, 0])
    plt.show()

    
