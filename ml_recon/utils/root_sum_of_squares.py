def root_sum_of_squares(data, coil_dim=0):
    """ Takes asquare root sum of squares of the abosolute value of complex data along the coil dimension

    Args:
        data (torch.Tensor): Data needed to be coil combined
        coil_dim (int, optional): dimension index. Defaults to 0.

    Returns:
        torch.Tensor: Coil combined data
    """
    assert data.ndims > coil_dim
    return data.abs().pow(2).sum(coil_dim).sqrt()
    