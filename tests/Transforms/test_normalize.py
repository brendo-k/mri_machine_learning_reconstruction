from ml_recon.Transforms import normalize
import torch


def test_normalize():
    x = torch.rand(10, 4, 640, 320)
    sample = {'k_space': x}
    norm = normalize()
    x_norm = norm(sample)
    x_max = x_norm['k_space'].amax(dim=(-1, -2))
    x_min = x_norm['k_space'].amin(dim=(-1, -2))
    assert (x_max == 1).all()
    assert (x_min == 0).all()

def test_no_batch_normalize():
    x = torch.rand(10, 640, 320)
    sample = {'k_space': x}
    norm = normalize()
    x_norm = norm(sample)
    x_max = x_norm['k_space'].amax(dim=(-1, -2))
    x_min = x_norm['k_space'].amin(dim=(-1, -2))
    assert (x_max == 1).all()
    assert (x_min == 0).all()

