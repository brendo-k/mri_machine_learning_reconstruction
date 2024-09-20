"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


import torch
from ml_recon.Loss.ssim_loss import SSIMLoss


def mse(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute Mean Squared Error (MSE)"""
    return ((gt - pred) ** 2).sum()/torch.sum(mask, dim=(-1, -2), keepdim=True)


def nmse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Normalized Mean Squared Error (NMSE)
    Don't need mask here because norms aren't affected by masking
    """
    nmse = torch.linalg.vector_norm(gt - pred, 2, dim=(-1, -2)) ** 2 / torch.linalg.vector_norm(gt, 2, dim=(-1, -2)) ** 2
    return nmse.mean()


def psnr(
    gt: torch.Tensor, pred: torch.Tensor, mask) -> torch.Tensor:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    error = mse(gt, pred, mask)
    psnr = 10 * torch.log10(pred.amax(dim=(-1, -2)).pow(2)/error)
    return psnr.mean()


def ssim(
        gt: torch.Tensor, pred: torch.Tensor, device, reduce=True, max_val: float = 0) -> torch.Tensor:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    if max_val == 0:
        max_val = gt.max().item()
    

    ssim_func = SSIMLoss().to(device)
    # subtract by 1 since ssimloss is inverted 
    ssim = ssim_func(
            gt, pred, data_range=max_val, reduced=reduce
        )

    return 1 - ssim
