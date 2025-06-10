import torch

class KMaxSoftmaxFunction(torch.autograd.Function):
    """Relaxation of the sampling operation. In the forward pass, the sampling 
    operation is discrete. However, in the backwards pass, we relax the gradients 
    to a sigmoid with a slope. This allows for backprop through discrete nodes

    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, slope: float):
        # Save input for backward pass
        ctx.save_for_backward(input)
        ctx.slope = slope

        # above zero locations are in the mask
        mask = (input >= 0).to(torch.float32)
        
        return mask
   
    @staticmethod
    # calculate gradient by relaxing to sigmoid (gradient=sig(x)(1-sig(x)*slope)
    def backward(ctx, grad_output): # type: ignore
        # Retrieve the saved input
        input, = ctx.saved_tensors
        
        # Apply softmax to the input
        sigmoid_output = torch.sigmoid(input*ctx.slope)
        
        # Compute gradient of softmax wrt input
        grad_input = grad_output * (sigmoid_output * (1 - sigmoid_output)) * ctx.slope
        
        return grad_input, None, None # None for k_percent, as it doesn't require gradient


