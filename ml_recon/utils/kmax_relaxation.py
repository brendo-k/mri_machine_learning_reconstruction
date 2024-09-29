import torch
from torch.functional import F

class KMaxSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor, k_percent: float, slope: float):
        # Save input for backward pass
        ctx.save_for_backward(input)
        ctx.slope = slope

        b, contrast, channel, h, w = input.shape
        
        # number of values to pass
        k = int(k_percent * input.numel())
        flattened_input = input.view(b, contrast, channel, h*w)
        _, indices = torch.topk(flattened_input, k, dim=-1)
        
        # Create a mask that zeros out the values not in the top k%
        mask = torch.zeros_like(flattened_input)
        mask[indices] = 1
        
        # Apply mask to the input
        output = flattened_input * mask
        return output.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input
        input, = ctx.saved_tensors
        
        # Apply softmax to the input
        softmax_output = torch.sigmoid(input*slope)
        
        # Compute gradient of softmax wrt input
        grad_input = grad_output * softmax_output
        
        return grad_input, None, None # None for k_percent, as it doesn't require gradient

