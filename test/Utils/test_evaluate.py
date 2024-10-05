import torch 
import pytest

from ml_recon.utils.evaluate import ssim

def test_ssim():
    x = torch.randn(4, 1, 320, 320)

    assert 1 == ssim(x, x, device='cpu')
