
def combine_coils(data, coil_dim=0):
    return data.pow(2).sum(coil_dim).sqrt()
    