import pytest
import numpy as np
import torch
from train_utils import to_device
from ml_recon.dataset.undersample import gen_pdf_columns, get_mask_from_distribution


def test_passthrough():
    data = (torch.rand(3, 16, 320, 320), torch.rand(3, 16, 320, 320), torch.rand(3, 16, 320, 320), torch.rand(3, 320, 320))
    mask, input, target, loss, zf = to_device(data, 'cpu', 'supervised')
    
    assert mask.ndim == 4
    assert input.ndim == 4
    assert target.ndim == 4
    assert loss.ndim == 4
    assert zf.ndim == 4

    assert mask.shape == (3, 16, 320, 320) 
    assert input.shape == (3, 16, 320, 320) 
    assert target.shape == (3, 16, 320, 320) 
    assert loss.shape == (3, 16, 320, 320) 
    assert zf.shape == (3, 16, 320, 320) 

def test_supervised():
    nx, ny = 256, 256
    prob = gen_pdf_columns(nx, ny, 1/4, 8, 10)
    prob2 = gen_pdf_columns(nx, ny, 1/2, 8, 10)
    rng = np.random.default_rng()
    mask = torch.from_numpy(get_mask_from_distribution(prob, rng, deterministic=True))
    mask2 = torch.from_numpy(get_mask_from_distribution(prob2, rng, deterministic=True))

    k_space = torch.rand(3, 16, 256, 256, dtype=torch.complex64) 
    under = mask * k_space
    doub_under = mask2 * k_space

    data = (doub_under, under, k_space, torch.rand(3, 256, 256))
    mask_input, input, target, loss, zf = to_device(data, 'cpu', 'supervised')

    mask = mask.tile(3, 16, 1, 1)

    torch.testing.assert_close(mask_input, mask.type(torch.bool))
    torch.testing.assert_close(target, k_space)
    torch.testing.assert_close(input, under)
    torch.testing.assert_close(loss, torch.ones_like(k_space))
    torch.testing.assert_close(zf, torch.ones_like(k_space).type(torch.bool))


def test_self():
    nx, ny = 256, 256
    prob = gen_pdf_columns(nx, ny, 1/4, 8, 10)
    prob2 = gen_pdf_columns(nx, ny, 1/2, 8, 10)
    rng = np.random.default_rng()
    mask = torch.from_numpy(get_mask_from_distribution(prob, rng, deterministic=True))
    mask2 = torch.from_numpy(get_mask_from_distribution(prob2, rng, deterministic=True))

    k_space = torch.rand(3, 16, 256, 256) 
    under = mask * k_space
    doub_under = mask2 * k_space

    data = (doub_under, under, k_space, torch.rand(3, 256, 256))
    mask_input, input, target, loss, zf = to_device(data, 'cpu', 'supervised')

    mask = mask.tile(3, 16, 1, 1)

    torch.testing.assert_close(mask_input, mask.type(torch.bool))
    torch.testing.assert_close(target, k_space)
    torch.testing.assert_close(input, under)
    torch.testing.assert_close(loss, torch.ones_like(k_space))
    torch.testing.assert_close(zf, torch.ones_like(k_space).type(torch.bool))





