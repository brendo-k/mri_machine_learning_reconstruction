import torch
from typing import Callable
def evaluate_over_contrasts(function: Callable, ground_truth:torch.Tensor, estimated:torch.Tensor):
    assert ground_truth.ndim == 4, 'should be 4 dimensional'
    assert estimated.ndim == 4, 'should be 4 dimensional'
    assert ground_truth.shape == estimated.shape

    metric_value = []
    for i in range(ground_truth.shape[1]):
        gt_contrast = ground_truth[:, [i], :, :]
        estimated_contrast = estimated[:, [i], :, :]
        metric_value.append(function(gt_contrast, estimated_contrast))

    return sum(metric_value)/len(metric_value)


        

