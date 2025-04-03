import pytest 
import torch 
from ml_recon.pl_modules.pl_learn_ssl_undersampling import (
    LearnedSSLLightning, 
    VarnetConfig, 
    DualDomainConifg, 
    LearnPartitionConfig,
    )
import numpy as np

IMAGE_SIZE = (2, 128, 128)

@pytest.fixture()
def build_model() -> LearnedSSLLightning:
    return LearnedSSLLightning(
        LearnPartitionConfig(IMAGE_SIZE, inital_R_value=2),
        varnet_config=VarnetConfig(['t1']),
        dual_domain_config=DualDomainConifg(
            is_pass_inverse=True, 
            is_pass_original=True,
            inverse_no_grad=False,
            original_no_grad=False,
            )
        )

@pytest.fixture()
def ssl_batch() -> dict:
    k_space = torch.randn((2, IMAGE_SIZE[0], 1, IMAGE_SIZE[1], IMAGE_SIZE[2]), dtype=torch.complex64)
    initial_mask = torch.rand((2, IMAGE_SIZE[0], 1, IMAGE_SIZE[1], IMAGE_SIZE[2])) > 0.5 
    initial_mask[..., IMAGE_SIZE[1]//2 - 5:IMAGE_SIZE[1]//2 - 5, IMAGE_SIZE[2]//2 - 5:IMAGE_SIZE[2]//2 - 5] = 1 
    second_mask = torch.rand((2, IMAGE_SIZE[0], 1, IMAGE_SIZE[1], IMAGE_SIZE[2])) > 0.5 
    second_mask[..., IMAGE_SIZE[1]//2 - 5:IMAGE_SIZE[1]//2 - 5, IMAGE_SIZE[2]//2 - 5:IMAGE_SIZE[2]//2 - 5] = 1 
    mask = initial_mask * second_mask
    loss_mask = initial_mask * (~second_mask)

    undersampled = k_space * initial_mask
    return {
        'fs_k_space': k_space, 
        'undersampled': undersampled,
        'mask': mask.to(torch.float32),
        'loss_mask': loss_mask.to(torch.float32)
    }

@pytest.fixture()
def supervised_batch() -> dict:
    k_space = torch.randn((2, IMAGE_SIZE[0], 1, IMAGE_SIZE[1], IMAGE_SIZE[2]), dtype=torch.complex64)
    initial_mask = torch.rand((2, IMAGE_SIZE[0], 1, IMAGE_SIZE[1], IMAGE_SIZE[2])) > 0.5 
    initial_mask[..., IMAGE_SIZE[1]//2 - 5:IMAGE_SIZE[1]//2 - 5, IMAGE_SIZE[2]//2 - 5:IMAGE_SIZE[2]//2 - 5] = 1 
    loss_mask = torch.ones_like(initial_mask)
    mask = initial_mask

    undersampled = k_space * initial_mask
    return {
        'fs_k_space': k_space, 
        'undersampled': undersampled,
        'mask': mask.to(torch.float32),
        'loss_mask': loss_mask.to(torch.float32)
    }

def test_partitioning_ssl_learned(build_model: LearnedSSLLightning, ssl_batch): 
    build_model.enable_learn_partitioning = True


    input_mask, loss_mask = build_model.partition_k_space(ssl_batch)
    initial_mask = ssl_batch['undersampled'] != 0

    torch.testing.assert_close(input_mask + loss_mask, initial_mask.to(torch.float32))
    assert input_mask.dtype == torch.float32
    assert loss_mask.dtype == torch.float32

def test_partitioning_ssl_unlearned(build_model: LearnedSSLLightning, ssl_batch): 
    build_model.enable_learn_partitioning = False

    input_mask, loss_mask = build_model.partition_k_space(ssl_batch)
    initial_mask = ssl_batch['undersampled'] != 0

    torch.testing.assert_close(input_mask + loss_mask, initial_mask.to(torch.float32))
    assert input_mask.dtype == torch.float32
    assert loss_mask.dtype == torch.float32


def test_supervised_partitioning(build_model: LearnedSSLLightning, supervised_batch): 
    build_model.enable_learn_partitioning = False

    input_mask, loss_mask = build_model.partition_k_space(supervised_batch)
    initial_mask = supervised_batch['undersampled'] != 0

    torch.testing.assert_close(input_mask, initial_mask.to(torch.float32))
    torch.testing.assert_close(loss_mask, torch.ones_like(input_mask))
    assert input_mask.dtype == torch.float32
    assert loss_mask.dtype == torch.float32
    
def test_triple_pathway(build_model: LearnedSSLLightning, supervised_batch): 
    build_model.enable_learn_partitioning = True

    input_mask, loss_mask = build_model.partition_k_space(supervised_batch)
    initial_mask = supervised_batch['undersampled'] != 0

    torch.testing.assert_close(input_mask, initial_mask.to(torch.float32))
    torch.testing.assert_close(loss_mask, torch.ones_like(input_mask))
    assert input_mask.dtype == torch.float32
    assert loss_mask.dtype == torch.float32