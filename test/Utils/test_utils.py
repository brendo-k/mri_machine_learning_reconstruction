from ml_recon.utils.root_sum_of_squares import combine_coils
import torch
import math

def test_combine_coils_unity():
    x = torch.zeros((1, 1, 1))
    x[0, 0, 0] = 5
    x_combined = combine_coils(x)
    assert x_combined.ndim == 2
    assert x_combined[0, 0] == 5


def test_combine_coils():
    x = torch.zeros((1, 2, 1))
    x[0, 0, 0] = 5
    x[0, 1, 0] = 2
    x_combined = combine_coils(x, coil_dim=1)
    assert x_combined.ndim == 2
    assert x_combined[0, 0] == math.sqrt(5**2 + 2**2)