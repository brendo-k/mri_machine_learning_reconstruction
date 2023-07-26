
def root_sum_of_squares(data, coil_dim=0):
    return data.abs().pow(2).sum(coil_dim).sqrt()
    