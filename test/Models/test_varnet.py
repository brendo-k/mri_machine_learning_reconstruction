from ml_recon.models.varnet import VarNet
import torch

def test_passthrough():
    x = torch.rand((2, 6, 640, 320))
    y = torch.full((2, 640, 320), True)
    sense_model = VarNet(2, 2)
    output = sense_model(x, y)

    assert output.shape == x.shape
