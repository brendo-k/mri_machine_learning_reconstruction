import numpy as np

def test_apply_sensetivites(brats_dataset):
    x = np.random.rand(5, brats_dataset.nx, brats_dataset.ny)
    x_sense = brats_dataset.apply_sensetivities(x) 

    assert x_sense.dtype == np.complex_
    assert x_sense.ndim == 4
    assert x_sense.shape == (5, 8, brats_dataset.nx, brats_dataset.ny)

def test_generate_phase(brats_dataset):
    x = np.random.rand(5, brats_dataset.nx, brats_dataset.ny) + 1j * np.random.rand(5, brats_dataset.nx, brats_dataset.ny)
    x_phase = brats_dataset.generate_and_apply_phase(x)
