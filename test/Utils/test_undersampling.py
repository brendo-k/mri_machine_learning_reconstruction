import numpy as np
import pytest
import torch

from ml_recon.utils.undersample_tools import (
    gen_pdf_columns, 
    gen_pdf_bern, 
    get_mask_from_distribution, 
    get_mask_from_segregated_sampling, 
    scale_pdf,
    ssdu_gaussian_selection
    )


@pytest.mark.parametrize("resolution", [[300, 600], [256, 256], [128, 128]])
def test_line_probability_mask(resolution):
    pdf = gen_pdf_columns(resolution[0], resolution[1], 1/8, 8, 10)
    
    torch.testing.assert_close(np.mean(pdf), 1/8)
    assert pdf.shape == (resolution[1], resolution[0])
    torch.testing.assert_close(pdf[:, resolution[0]//2 - 5: resolution[0]//2 + 5], np.ones((resolution[1], 10)))

def test_bern_2d():
    
    pdf = gen_pdf_bern(120, 320, 1/8, 8, 10)
    
    assert pdf.shape == (320, 120)
    torch.testing.assert_close(pdf[320//2-5:320//2+5, 120//2 - 5: 120//2 + 5], np.ones((10, 10)))
    assert 1/8, torch.from_numpy(pdf).mean().item()

def test_bern_segregated():
    
    pdf = gen_pdf_bern(120, 320, 1/8, 8, 10)
    pdf = np.repeat(pdf[np.newaxis, :, :], 4, axis=0)
    rng = np.random.default_rng()

    masks, probs = get_mask_from_segregated_sampling(pdf, rng, line_constrained = False)
    assert masks.shape == pdf.shape
    
def test_scaling():
    
    pdf_R8 = gen_pdf_bern(120, 320, 1/8, 8, 10)
    pdf_R4 = gen_pdf_bern(120, 320, 1/4, 8, 10)
    
    scaled_pdf = scale_pdf(pdf_R8, 4, 10)

    np.testing.assert_allclose(scaled_pdf, pdf_R4) 
    
def test_scaling_multiple_dims():
    
    pdf_R8 = gen_pdf_bern(120, 320, 1/8, 8, 10)
    pdf_R8 = np.tile(pdf_R8[np.newaxis, :, :], (6, 1, 1))
    pdf_R8_copy = pdf_R8.copy()
    pdf_R4 = gen_pdf_bern(120, 320, 1/4, 8, 10)
    pdf_R4 = np.tile(pdf_R4[np.newaxis, :, :], (6, 1, 1))
    
    scaled_pdf = scale_pdf(pdf_R8, 4, 10)

    np.testing.assert_allclose(scaled_pdf, pdf_R4) 

    #ensure no inplace operations took place
    np.testing.assert_allclose(pdf_R8, pdf_R8_copy) 
    
def test_ssdu_selection():
    mask = np.random.randn(128, 128) > 0
    input, loss = ssdu_gaussian_selection(mask)

    torch.testing.assert_close(input ^ loss, mask)