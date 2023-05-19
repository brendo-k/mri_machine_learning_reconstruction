import numpy as np

def normalized_mse(x, y):
    return np.linalg.norm(x - y).pow(2)/np.linalg.norm(y).pow(2)

def normalize(vect):
    std = np.std(vect)
    mean = np.mean(vect)
    return (vect - mean)/std