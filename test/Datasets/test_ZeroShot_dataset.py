import pytest
from ml_recon.dataset.Zeroshot_datset import ZeroShotDataset


def test_passthrough():
    dataset1 = ZeroShotDataset('../test_data/simulated_subset_random_phase/train/BraTS2021_00017/BraTS2021_00017.h5', validation=True)
    dataset2 = ZeroShotDataset('../test_data/simulated_subset_random_phase/train/BraTS2021_00017/BraTS2021_00017.h5', validation=False)