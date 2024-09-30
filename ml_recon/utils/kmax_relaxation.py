import torch

class KMaxSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, k_percent: torch.Tensor, slope: float):
        # Save input for backward pass
        ctx.save_for_backward(input)
        ctx.slope = slope

        b, contrast, h, w = input.shape

        # Calculate the number of values to pass for each contrast
        k = (1 / k_percent * h * w).to(torch.int)

        # Flatten the input to operate over the spatial dimensions
        flattened_input = input.view(b, contrast, h * w)

        # Create a mask initialized to zeros
        mask = torch.zeros_like(flattened_input)

        # Vectorized loop over the contrast dimension
        for c in range(contrast):
            # Get top-k indices for each contrast, batch-wise
            topk_vals, topk_indices = torch.topk(flattened_input[:, c, :], k[c].item(), dim=-1, largest=True)
            
            # Set the mask for top k indices for each batch in the contrast
            mask[torch.arange(b).unsqueeze(1), c, topk_indices] = 1

        # Reshape the mask to the original input shape and return
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

