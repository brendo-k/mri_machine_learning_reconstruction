import pytest 
import torch 
from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning, VarnetConfig, DualDomainConifg, LearnPartitionConfig
import numpy as np

IMAGE_SIZE = (2, 128, 128)

@pytest.fixture()
def build_model() -> LearnedSSLLightning:
    return LearnedSSLLightning(
        LearnPartitionConfig(IMAGE_SIZE, inital_R_value=2),
        varnet_config=VarnetConfig(['t1']),
        dual_domain_config=DualDomainConifg(True, True)
        )

@pytest.fixture()
def batch() -> dict:
    k_space = torch.randn((2, IMAGE_SIZE[0], 1, IMAGE_SIZE[1], IMAGE_SIZE[2]), dtype=torch.complex64)
    initial_mask = torch.rand((2, IMAGE_SIZE[0], 1, IMAGE_SIZE[1], IMAGE_SIZE[2])) > 0.5 
    initial_mask[..., IMAGE_SIZE[1]//2 - 5:IMAGE_SIZE[1]//2 - 5, IMAGE_SIZE[2]//2 - 5:IMAGE_SIZE[2]//2 - 5] = 1 
    second_mask = torch.rand((2, IMAGE_SIZE[0], 1, IMAGE_SIZE[1], IMAGE_SIZE[2])) > 0.5 
    second_mask[..., IMAGE_SIZE[1]//2 - 5:IMAGE_SIZE[1]//2 - 5, IMAGE_SIZE[2]//2 - 5:IMAGE_SIZE[2]//2 - 5] = 1 
    undersampled = k_space * initial_mask
    return {
        'fs_k_space': k_space, 
        'undersampled': undersampled,
        'mask': initial_mask.to(torch.float32),
        'loss_mask': second_mask.to(torch.float32)
    }

def test_partitioning_ssl_learned(build_model: LearnedSSLLightning, batch): 
    build_model.is_learn_partitioning = True


    input_mask, loss_mask = build_model.partition_k_space(batch)
    initial_mask = batch['undersampled'] != 0

    torch.testing.assert_close(input_mask + loss_mask, initial_mask)
    assert input_mask.dtype == torch.float32
    assert loss_mask.dtype == torch.float32

def test_partitioning_unlearned(build_model: LearnedSSLLightning, batch): 
    build_model.is_learn_partitioning = False

    input_mask, loss_mask = build_model.partition_k_space(batch)
    initial_mask = batch['undersampled'] != 0

    torch.testing.assert_close(input_mask + loss_mask, initial_mask)
    assert input_mask.dtype == torch.float32
    assert loss_mask.dtype == torch.float32