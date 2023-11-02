import torch 
import pytest

from ml_recon.utils.evaluate import ssim

def test_ssim():
    x = torch.randn(4, 320, 320)

    assert 1 == ssim(x, x)
    assert x.shape == 1

    y = torch.randn(4, 1, 320, 320)

    with pytest.raises(ValueError) as execinfo:
        ssim(x, y)
    assert str(execinfo.value) == "Unexpected number of dimensions in pred."

    with pytest.raises(ValueError) as execinfo:
        ssim(y, x)
    assert str(execinfo.value) == "Unexpected number of dimensions in ground truth."

