import numpy as np
from typing import Union
import numpy.typing as npt
import torch

def root_sum_of_squares(data: Union[torch.Tensor, npt.NDArray[np.float_]], coil_dim=0):
    """ Takes asquare root sum of squares of the abosolute value of complex data along the coil dimension

    Args:
        data (torch.Tensor): Data needed to be coil combined
        coil_dim (int, optional): dimension index. Defaults to 0.

    Returns:
        torch.Tensor: Coil combined data
    """
    assert data.ndim > coil_dim
    if isinstance(data, np.ndarray):
        return np.sqrt(np.sum(np.power(np.abs(data), 2), axis=coil_dim))
    elif isinstance(data, torch.Tensor):
        return torch.sqrt(data.abs().pow(2).sum(coil_dim) + 1e-6)
    else:
        raise ValueError(f'Data should be either a numpy array or pytorch Tensor')
    
