import pytest
import torch

from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule

BATCH_SIZE = 2
RESOLUTION = (128, 128)

@pytest.fixture
def define_datamodule():
    data_module = UndersampledDataModule(
            'brats', 
            'test/test_data/simulated_subset_random_phase', 
            batch_size=BATCH_SIZE, 
            resolution=RESOLUTION,
            num_workers=0,
            contrasts=['t1', 't2', 't1ce', 'flair'],
            line_constrained=False, 
            self_supervsied=False,
            is_variable_density=True,
            R=4
            ) 
    data_module.setup('train')
    return data_module


def test_datasetIntegration(define_datamodule):
    data_module = define_datamodule

    train_batch = next(iter(data_module.train_dataloader()))
    val_batch = next(iter(data_module.train_dataloader()))
    test_batch = next(iter(data_module.train_dataloader()))

    undersampled = train_batch['input']
    undersampled_val = val_batch['input']
    undersampled_test = test_batch['input']
    average_undersampling = (undersampled != 0).to(torch.float32).mean((-1, -2))
    average_undersampling_val = (undersampled_val != 0).to(torch.float32).mean((-1, -2))
    average_undersampling_test = (undersampled_test != 0).to(torch.float32).mean((-1, -2))

    assert undersampled.ndim == 5
    assert undersampled.shape == (BATCH_SIZE, 4, 10, ) + RESOLUTION
    torch.testing.assert_close(average_undersampling, torch.full((BATCH_SIZE, 4, 10), 1/4), atol=0.01, rtol=0)

    assert undersampled_val.ndim == 5
    assert undersampled_val.shape == (BATCH_SIZE, 4, 10, ) + RESOLUTION
    torch.testing.assert_close(average_undersampling_val, torch.full((BATCH_SIZE, 4, 10), 1/4), atol=0.01, rtol=0)

    assert undersampled_test.ndim == 5
    assert undersampled_test.shape == (BATCH_SIZE, 4, 10, ) + RESOLUTION
    torch.testing.assert_close(average_undersampling_test, torch.full((BATCH_SIZE, 4, 10), 1/4), atol=0.01, rtol=0)


def test_datasetScaling(define_datamodule):
    data_module = define_datamodule

    train_batch = next(iter(data_module.train_dataloader()))
    val_batch = next(iter(data_module.train_dataloader()))
    test_batch = next(iter(data_module.train_dataloader()))

    undersampled = train_batch['input']
    initial_mask = undersampled != 0
    fully_sampled = train_batch['fs_k_space']
    target = train_batch['target']

    torch.testing.assert_close(undersampled, initial_mask*fully_sampled)
    torch.testing.assert_close(undersampled, initial_mask * target)
    torch.testing.assert_close(
        undersampled.abs().amax((-1, -2, -3)),  
        torch.ones(undersampled.shape[:2]))

