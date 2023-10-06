"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional

import torch
import numpy as np
from ml_recon.Loss.ssim_loss import SSIMLoss


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute Mean Squared Error (MSE)"""
    return ((gt - pred) ** 2).mean()


def nmse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return torch.linalg.vector_norm(gt - pred) ** 2 / torch.linalg.vector_norm(gt) ** 2


def psnr(
    gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    error = mse(gt, pred)
    return 10 * torch.log10(pred.max().pow(2)/error)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = SSIMLoss()(
            gt, pred, data_range=maxval
        )

    return ssim 
