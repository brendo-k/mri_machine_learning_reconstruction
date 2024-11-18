"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        reduced: bool = True,
    ):
        assert isinstance(self.w, torch.Tensor)

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced:
            return 1 - S.mean()
        else:
            return 1 - S

class L1L2Loss():
    def __init__(self, norm_all_k):
        self.norm_all_k = norm_all_k
        
    def __call__(self, target: torch.Tensor, predicted: torch.Tensor):
        assert not torch.isnan(target).any()
        assert not torch.isnan(predicted).any()
        target = torch.view_as_complex(target)
        predicted = torch.view_as_complex(predicted)
        norm_dims = (3, 4)
        if self.norm_all_k:
            norm_dims = (2, ) + norm_dims
        l2_component = torch.linalg.vector_norm(target - predicted, 2, dim=norm_dims)
        l1_component = torch.linalg.vector_norm(target - predicted, 1, dim=norm_dims)
        l2_norm = (torch.linalg.vector_norm(target, 2, dim=norm_dims) + 1e-20)
        l1_norm = (torch.linalg.vector_norm(target, 1, dim=norm_dims) + 1e-20)

        if torch.isnan(l2_component).any():
            print(target)
        
        loss  = torch.sum(l2_component/l2_norm + l1_component/l1_norm)/target.numel()
        return loss

class L1ImageGradLoss():
    def __init__(self, grad_scaling):
        self.grad_scaling = grad_scaling
    
    def __call__(self, targ, pred):
        l1_loss = torch.nn.L1Loss()
        grad_x = lambda targ, pred: l1_loss(targ.diff(dim=-1), pred.diff(dim=-1))
        grad_y = lambda targ, pred: l1_loss(targ.diff(dim=-2), pred.diff(dim=-2))

        return l1_loss(targ, pred) + self.grad_scaling(grad_x(targ, pred) + grad_y(targ, pred))