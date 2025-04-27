"""
"""

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing import Union, Literal, Optional, Sequence, Any


class L1L2Loss(torch.nn.Module):
    def __init__(
        self, 
        norm_all_k: bool, 
        reduce_mean: bool = True
    ):
        self.norm_all_k = norm_all_k
        self.reduce_mean = reduce_mean
        
    def forward(self, target: torch.Tensor, predicted: torch.Tensor):
        assert not torch.isnan(target).any()
        assert not torch.isnan(predicted).any()
        target = torch.view_as_complex(target)
        predicted = torch.view_as_complex(predicted)

        norm_dims = (3, 4)
        if self.norm_all_k:
            norm_dims = (2,) + norm_dims

        l2_component = torch.linalg.vector_norm(target - predicted, 2, dim=norm_dims)
        l1_component = torch.linalg.vector_norm(target - predicted, 1, dim=norm_dims)
        l2_norm = (torch.linalg.vector_norm(target, 2, dim=norm_dims) + 1e-20)
        l1_norm = (torch.linalg.vector_norm(target, 1, dim=norm_dims) + 1e-20)

        loss  = torch.sum(l2_component/l2_norm + l1_component/l1_norm)
        if self.reduce_mean:
            loss /= predicted.numel()
        
        return loss

class L1ImageGradLoss(torch.nn.Module):
    def __init__(self, grad_scaling):
        self.grad_scaling = grad_scaling
    
    def forward(self, targ, pred):
        l1_loss = torch.nn.L1Loss()
        grad_x = l1_loss(targ.diff(dim=-1), pred.diff(dim=-1))
        grad_y = l1_loss(targ.diff(dim=-2), pred.diff(dim=-2))

        return l1_loss(targ, pred) + self.grad_scaling * (grad_x + grad_y)


class SSIM_Loss(StructuralSimilarityIndexMeasure):
    """
    Torchmetrics SSIM class. Inverted to calculate loss
    """
    def __init__(
        self,
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        data_range: Optional[Union[float, tuple[float, float]]] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        return_full_image: bool = False,
        return_contrast_sensitivity: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            gaussian_kernel,
            sigma, 
            kernel_size, 
            reduction,
            data_range,
            k1, 
            k2, 
            return_full_image, 
            return_contrast_sensitivity,
            **kwargs
        )

    def forward(self, pred, target): 
        return 1 - super().forward(pred, target)


