import torch
from typing import Callable

def evaluate_over_contrasts(function: Callable, ground_truth:torch.Tensor, estimated:torch.Tensor):
    assert ground_truth.shape == estimated.shape

    metric_value = []
    for i in range(ground_truth.shape[1]):
        contrast_metric_value = 0
        for j in range(ground_truth.shape[0]):
            gt_contrast = ground_truth[j, i, ...].unsqueeze(0).unsqueeze(0)
            estimated_contrast = estimated[j, i, ...].unsqueeze(0).unsqueeze(0)
                # Call without `data_range` argument
            contrast_metric_value += (function(gt_contrast, estimated_contrast))
        metric_value.append(contrast_metric_value/ground_truth.shape[0])

    return metric_value


        

