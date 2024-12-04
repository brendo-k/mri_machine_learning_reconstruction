import torch

class KMaxSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, slope: float):
        # Save input for backward pass
        ctx.save_for_backward(input)
        ctx.slope = slope

        mask = (input > 0).to(torch.float32)
        
        # Reshape the mask to the original input shape and return
        return mask
   
    @staticmethod
    def backward(ctx, grad_output): # type: ignore
        # Retrieve the saved input
        input, = ctx.saved_tensors
        
        # Apply softmax to the input
        sigmoid_output = torch.sigmoid(input*ctx.slope)
        
        # Compute gradient of softmax wrt input
        grad_input = grad_output * (sigmoid_output * (1 - sigmoid_output)) * ctx.slope
        
        return grad_input, None, None # None for k_percent, as it doesn't require gradient


class KMaxSoftmaxFunction1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor, k_percent: float, slope: float):
        # Save input for backward pass
        ctx.save_for_backward(input)
        ctx.slope = slope

        b, contrast, channel, h, w = input.shape
        input = input.permute((0, 1, 2, 4, 3))
        
        # number of values to pass
        k = int(k_percent * w)
        _, indices = torch.topk(input, k, dim=-1)
        
        # Create a mask that zeros out the values not in the top k%
        mask = torch.zeros_like(input)
        mask[indices] = 1
        
        return mask.view_as(input)

    @staticmethod
    def backward(ctx, grad_output): # type: ignore
        # Retrieve the saved input
        input, = ctx.saved_tensors
        
        # Apply softmax to the input
        softmax_output = torch.sigmoid(input*ctx.slope)
        
        # Compute gradient of softmax wrt input
        grad_input = grad_output * softmax_output
        
        return grad_input, None, None # None for k_percent, as it doesn't require gradient

