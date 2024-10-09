from ml_recon.utils.root_sum_of_squares import root_sum_of_squares
import torch
import math

def test_combine_coils_unity():
    x = torch.zeros((1, 1, 1))
    x[0, 0, 0] = 5
    x_combined = root_sum_of_squares(x)
    assert x_combined.ndim == 2
    assert x_combined[0, 0] == 5


def test_combine_coils():
    x = torch.zeros((3, 2, 128, 128))
    x[0, 0, 0, 0] = 5
    x[0, 1, 0, 0] = 2
    x[0, 0, 100, 100] = 5
    x[0, 1, 100, 100] = 5
    x_combined = root_sum_of_squares(x, coil_dim=1)
    assert x_combined.ndim == 3

    result = torch.full((3, 128, 128), math.sqrt(1e-20))  #zero locations should be sqrt(1e-6) (can't be zero)
    result[0, 0, 0] = math.sqrt(5**2 + 2**2)
    result[0, 100, 100] = math.sqrt(5**2 + 5**2)
    torch.testing.assert_close(x_combined, result)