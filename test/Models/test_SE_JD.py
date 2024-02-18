import pytest
from ml_recon.models.SE_JD import SingleEncoderJointDecoder
import torch

def test_init(): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SingleEncoderJointDecoder(8, 4, 4, 32, 4, drop_prob=0)
    model.to(device)
    print(sum([param.numel() for param in model.parameters()]))

    x = torch.randn(1, 8, 256, 256, dtype=torch.float32).to(device)

    assert x.shape == model(x).shape

