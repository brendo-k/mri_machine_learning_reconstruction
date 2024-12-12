# test/conftest.py
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml_recon')))


@pytest.fixture
def image_input_3_contrast():
    return torch.randn((2, 6, 128, 128), dtype=torch.float32)

@pytest.fixture
def device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device