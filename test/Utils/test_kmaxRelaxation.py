import torch 
import pytest
from ml_recon.utils.kmax_relaxation import KMaxSoftmaxFunction



def test_passthrough():
    kmax = KMaxSoftmaxFunction()
    b, con, h, w, = 2, 4, 128, 128
    R = 2
    slope = 100
    R_values = torch.full(size=(con,), fill_value=R)
    activations = torch.randn(b, con, h, w, dtype=torch.float32)
    mask = kmax.apply(activations, R_values, slope)

    average_mask = mask.mean((-1, -2)) 
    real_ground_truth = torch.full_like(average_mask, 1/2)

    # average values should be R
    torch.testing.assert_close(average_mask, real_ground_truth)
    assert mask.max() <= 1 and mask.min() >= 0 # constrained between 0 1 
    assert torch.all((mask == 1) | (mask == 0)) # constrained to be 0 or 1 

    # no masks are the same
    for i in range(b*con):
        for j in range(i+1, b*con):
            b_index1 = i//con
            con_index1 = i%con
            b_index2 = j//con
            con_index2 = j%con
            assert not torch.equal(mask[b_index1, con_index1, :, :], mask[b_index2, con_index2, :, :])


def test_backwards():
    kmax = KMaxSoftmaxFunction()
    b, con, h, w, = 2, 4, 128, 128
    R = 2
    slope = 100
    R_values = torch.full(size=(con,), fill_value=R)
    activations = torch.randn(b, con, h, w, dtype=torch.float32, requires_grad=True)

    mask = kmax.apply(activations, R_values, slope)


    sigmoid_output = torch.sigmoid(activations * slope)
    gradient = sigmoid_output * (1 - sigmoid_output) * slope

    # calculate gradients
    mask.sum().backward()

    torch.testing.assert_close(gradient, activations.grad)

    activations.grad.zero_()
    sigmoid_output.sum().backward()

    torch.testing.assert_close(gradient, activations.grad)
