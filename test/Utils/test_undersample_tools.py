import numpy as np
import pytest
import torch

from ml_recon.utils.undersample_tools import (
    gen_pdf_columns, 
    gen_pdf_bern, 
    get_mask_from_distribution, 
    apply_undersampling_from_dist,
    ssdu_gaussian_selection
    )


@pytest.mark.parametrize("resolution", [[300, 600], [256, 256], [128, 128]])
def test_line_probability_mask(resolution):
    pdf = gen_pdf_columns(resolution[0], resolution[1], 1/8, 8, 10)
    
    torch.testing.assert_close(np.mean(pdf), 1/8)
    assert pdf.shape == (resolution[1], resolution[0])
    torch.testing.assert_close(pdf[:, resolution[0]//2 - 5: resolution[0]//2 + 5], np.ones((resolution[1], 10)))


@pytest.mark.parametrize("acceleration_factor", [2, 4, 6, 8])
def test_bern_2d(acceleration_factor):
    
    pdf = gen_pdf_bern(320, 120, 1/acceleration_factor, 8, 10)
    
    assert pdf.shape == (320, 120)
    torch.testing.assert_close(pdf[320//2-5:320//2+5, 120//2 - 5: 120//2 + 5], np.ones((10, 10)))
    assert 1/acceleration_factor, torch.from_numpy(pdf).mean().item()
    assert pdf.max() <= 1
    assert pdf.min() >= 0
    

@pytest.mark.parametrize("acceleration_factor", [2, 4, 6, 8])
def test_columns(acceleration_factor):
    
    pdf = gen_pdf_columns(120, 320, 1/acceleration_factor, 8, 10)
    
    assert pdf.shape == (320, 120)
    torch.testing.assert_close(pdf[:, 120//2 - 5: 120//2 + 5], np.ones((320, 10)))
    assert 1/acceleration_factor, torch.from_numpy(pdf).mean().item()
    assert pdf.max() <= 1
    assert pdf.min() >= 0


    
def test_ssdu_selection():
    mask = (np.random.randn(1, 128, 128) > 0).astype(np.float32)
    input, loss = ssdu_gaussian_selection(mask)

    torch.testing.assert_close((input + loss)[:, 0, :, :], mask)


def test_apply_undersampling_not_same():
    rng = np.random.default_rng()
    k_space = rng.random(size=(4, 6, 128, 128)) + 1j * rng.random(size=(4, 6, 128, 128))
    pdf_R4 = gen_pdf_bern(128, 128, 1/4, 8, 10)
    pdf_R4 = np.stack([pdf_R4 for _ in range(k_space.shape[0])])

    undersampled_k, mask = apply_undersampling_from_dist(0, pdf_R4, k_space)

    k_mask = undersampled_k != 0 
    np.testing.assert_allclose(k_mask[:, [0], ...], mask)
    assert not np.all(undersampled_k == k_space)


def test_ssdu_gaussian_selection_shape():
    input_mask = np.ones((1, 64, 64), dtype=np.float32)
    trn_mask, loss_mask = ssdu_gaussian_selection(input_mask)

    input_mask = input_mask[:, None, :, :]
    assert trn_mask.shape == input_mask.shape
    assert loss_mask.shape == input_mask.shape

def test_ssdu_gaussian_selection_no_overlap():
    input_mask = np.ones((1, 64, 64), dtype=np.float32)
    trn_mask, loss_mask = ssdu_gaussian_selection(input_mask)
    assert np.all((trn_mask * loss_mask) == 0)

def test_ssdu_gaussian_selection_rho():
    input_mask = np.ones((1, 64, 64), dtype=np.float32)
    rho = 0.4
    trn_mask, loss_mask = ssdu_gaussian_selection(input_mask, rho=rho)
    expected_loss_count = int(np.ceil(np.sum(input_mask) * rho))
    actual_loss_count = np.sum(loss_mask)
    assert expected_loss_count == actual_loss_count, f"Expected {expected_loss_count} loss pixels, got {actual_loss_count}"

def test_ssdu_gaussian_selection_acs_region():
    input_mask = np.ones((1, 64, 64), dtype=np.float32)
    trn_mask, loss_mask = ssdu_gaussian_selection(input_mask)
    center_kx = input_mask.shape[2] // 2
    center_ky = input_mask.shape[1] // 2
    acs_shape = 10

    center_kx = slice(center_kx-acs_shape//2,center_kx+acs_shape//2)
    center_ky = slice(center_ky-acs_shape//2,center_ky+acs_shape//2)
    acs_region = loss_mask[:, center_kx, center_ky] 
    assert np.all(acs_region == 0) 
