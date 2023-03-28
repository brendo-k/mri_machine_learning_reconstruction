import numpy as np

def normalized_mse(x, y):
    return normalize(x - y)/normalize(y)

def normalize(vect):
    std = np.std(vect)
    mean = np.mean(vect)
    return (vect - mean)/std