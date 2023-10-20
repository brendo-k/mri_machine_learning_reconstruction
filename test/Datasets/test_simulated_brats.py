import numpy as np
import matplotlib.pyplot as plt

from ml_recon.dataset.Brats_dataset import SimulatedBrats
from ml_recon.utils import image_slices, ifft_2d_img

def test_apply_sensetivites():
    nx, ny = 256, 256
    x = np.random.rand(5, nx, ny)
    x_sense = SimulatedBrats.apply_sensetivities(x) 

    assert np.iscomplex(x_sense).any()
    assert x_sense.ndim == 4
    assert x_sense.shape == (5, 8, nx, ny)

def test_generate_phase():
    nx, ny = 256, 256
    x = np.random.rand(5, 8, nx, ny)
    x_phase = SimulatedBrats.generate_and_apply_phase(x, None)

    assert np.iscomplex(x_phase).all()
    assert (np.angle(x_phase) > 0.5).any() # ensure we have phase
    assert x_phase.shape == (5, 8, nx, ny)


