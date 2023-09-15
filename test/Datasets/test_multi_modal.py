import torch
import numpy as np
import pytest
import os

from ml_recon.dataset.multicontrast_loader import MultiContrastLoader

@pytest.fixture
def build_dataset() -> MultiContrastLoader:
    torch.manual_seed(0)
    dataset = MultiContrastLoader('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/training_subset/')
    return dataset

def test_init(build_dataset: MultiContrastLoader):
    data_list = build_dataset.data_list[0]
    keys = data_list.keys()
    assert 'flair' in keys
    assert 't1' in keys
    assert 't1ce' in keys
    assert 't2' in keys

    dir_names = [os.path.dirname(file) for file in  data_list.values()]
    # should all be the same values
    assert len(set(dir_names)) == 1

def test_apply_sensetivites(build_dataset: MultiContrastLoader):
    x = np.random.rand(5, 128, 128)
    x_sense = build_dataset.apply_sensetivities(x) 

    assert x_sense.dtype == np.complex_
    assert x_sense.ndim == 4
    assert x_sense.shape == (5, 8, 128, 128)

def test_generate_phase(build_dataset: MultiContrastLoader):
    x = np.random.rand(5, 128, 128) + 1j * np.random.rand(5, 128, 128)
    x_phase = build_dataset.generate_and_apply_phase(x)