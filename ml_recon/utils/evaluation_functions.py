"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


import torch
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_torch


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute Mean Squared Error (MSE)"""
    return ((gt - pred) ** 2).mean()


def nmse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Normalized Mean Squared Error (NMSE)
    Don't need mask here because norms aren't affected by masking
    """
    nmse = torch.linalg.vector_norm(gt - pred, 2, dim=(-1, -2)) ** 2 / torch.linalg.vector_norm(gt, 2, dim=(-1, -2)) ** 2
    return nmse.mean()


def psnr(
    gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    error = mse(gt, pred)
    psnr = 10 * torch.log10(pred.amax(dim=(-1, -2)).pow(2)/error)
    return psnr.mean()

