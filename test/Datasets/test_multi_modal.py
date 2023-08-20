import torch
import pytest

from ml_recon.dataset.multicontrast_loader import MultiContrastLoader

@pytest.fixture
def build_dataset() -> MultiContrastLoader:
    torch.manual_seed(0)
    dataset = MultiContrastLoader('/home/kadotab/projects/def-mchiew/kadotab/Datasets/Brats_2021/brats/training_data/')
    return dataset

def test_init(build_dataset: MultiContrastLoader):
    assert all(k in build_dataset.data_list[0].keys() for k in ('flair', 'T1', 'T1ce', 'T2'))