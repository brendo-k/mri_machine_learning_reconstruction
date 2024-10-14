import torch
from typing import Callable
import inspect 

def evaluate_over_contrasts(function: Callable, ground_truth:torch.Tensor, estimated:torch.Tensor):
    assert ground_truth.shape == estimated.shape

    function_params = inspect.signature(function).parameters
    accepts_data_range = 'data_range' in function_params

    metric_value = []
    for i in range(ground_truth.shape[1]):
        for j in range(ground_truth.shape[0]):
            gt_contrast = ground_truth[j, i, ...].unsqueeze(0).unsqueeze(0)
            estimated_contrast = estimated[j, i, ...].unsqueeze(0).unsqueeze(0)
            if accepts_data_range:
                # Call with `data_range` argument if the function accepts it
                max_val = max(gt_contrast.max().item(), estimated_contrast.max().item()) 
                min_val = min(gt_contrast.min().item(), estimated_contrast.min().item())
                metric_value.append(function(gt_contrast, estimated_contrast, data_range=(min_val, max_val)))
            else:
                # Call without `data_range` argument
                metric_value.append(function(gt_contrast, estimated_contrast))

    return sum(metric_value)/len(metric_value)


        

