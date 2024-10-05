import torch
from typing import Callable
import inspect 

def evaluate_over_contrasts(function: Callable, ground_truth:torch.Tensor, estimated:torch.Tensor):
    assert ground_truth.shape == estimated.shape

    function_params = inspect.signature(function).parameters
    accepts_data_range = 'data_range' in function_params

    metric_value = []
    for i in range(ground_truth.shape[1]):
        gt_contrast = ground_truth[:, [i], ...]
        estimated_contrast = estimated[:, [i], ...]
        if accepts_data_range:
            # Call with `data_range` argument if the function accepts it
            data_range = (gt_contrast - estimated_contrast).max() - (gt_contrast - estimated_contrast).min()
            metric_value.append(function(gt_contrast, estimated_contrast, data_range=data_range))
        else:
            # Call without `data_range` argument
            metric_value.append(function(gt_contrast, estimated_contrast))

    return sum(metric_value)/len(metric_value)


        

