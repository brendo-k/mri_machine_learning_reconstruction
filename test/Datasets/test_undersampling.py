import numpy as np
import torch
import pytest

from ml_recon.dataset.undersample import gen_pdf_columns

def test_probability_mask():
    pdf = gen_pdf_columns(300, 600, 1/4, 8, 10)
    
    torch.testing.assert_allclose(np.mean(pdf), 1/4)
    assert pdf.shape == (600, 300)
    torch.testing.assert_allclose(pdf[:, 150 - 5: 150 + 5], np.ones((600, 10)))