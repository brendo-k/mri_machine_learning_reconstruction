import numpy as np

def combine_coils(data, coil_dim=0):

    data_squared = np.square(data)
    data_summed = np.sum(data_squared, axis=coil_dim)
    return np.sqrt(data_summed)
    