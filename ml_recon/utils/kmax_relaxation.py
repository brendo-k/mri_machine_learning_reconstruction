import torch

class KMaxSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor, k_percent: float, slope: float):
        # Save input for backward pass
        ctx.save_for_backward(input)
        ctx.slope = slope

        b, contrast, h, w = input.shape
        
        # number of values to pass
        k = int(1/k_percent * h*w)
        flattened_input = input.view(b, contrast, h*w)
        _, indices = torch.topk(flattened_input, k, dim=-1, largest=True)
        
        # Create a mask that zeros out the values not in the top k%
        mask = torch.zeros_like(flattened_input)
        mask.scatter_(-1, indices, 1)  # Set the mask to 1 at the top n positions

        
        # Apply mask to the input
        output =  mask
        return output.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input
        input, = ctx.saved_tensors
        
        # Apply softmax to the input
        softmax_output = torch.sigmoid(input*ctx.slope)
        
        # Compute gradient of softmax wrt input
        grad_input = grad_output * softmax_output
        
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
    def backward(ctx, grad_output):
        # Retrieve the saved input
        input, = ctx.saved_tensors
        
        # Apply softmax to the input
        softmax_output = torch.sigmoid(input*ctx.slope)
        
        # Compute gradient of softmax wrt input
        grad_input = grad_output * softmax_output
        
        return grad_input, None, None # None for k_percent, as it doesn't require gradient

