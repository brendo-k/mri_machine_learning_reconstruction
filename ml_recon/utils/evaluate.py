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
    return ((gt - pred) ** 2).mean(dim=(-1, -2))


def nmse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute Normalized Mean Squared Error (NMSE)"""
    nmse = torch.linalg.vector_norm(gt - pred, 2, dim=(-1, -2)) ** 2 / torch.linalg.vector_norm(gt, 2, dim=(-1, -2)) ** 2
    return nmse.mean()


def psnr(
    gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    error = mse(gt, pred)
    psnr = 10 * torch.log10(pred.amax(dim=(-1, -2)).pow(2)/error)
    return psnr.mean()


def ssim(
    gt: torch.Tensor, pred: torch.Tensor, device) -> torch.Tensor:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() 

    ssim_func = SSIMLoss().to(device)
    # subtract by 1 since ssimloss is inverted 
    ssim = ssim_func(
            gt, pred, data_range=maxval
        )

    return 1 - ssim
