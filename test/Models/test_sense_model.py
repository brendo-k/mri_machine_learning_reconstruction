from ml_recon.Models.SensetivityModel import SensetivityModel
import torch

def test_passthrough():
    x = torch.rand((2, 6, 640, 320))
    sense_model = SensetivityModel(2, 2, 4)
    output = sense_model(x)

    assert output.shape == x.shape
