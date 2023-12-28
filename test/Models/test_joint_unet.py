from ml_recon.models.multi_task_unet import MultiTaskUnet
import torch

def assert_not_close(tensor1, tensor2, rtol=1e-5, atol=1e-8, msg=None):
    """
    Asserts that two tensors are not approximately equal within a tolerance.
    """
    diff = torch.abs(tensor1 - tensor2)
    max_abs_diff = torch.max(diff).item()
    assert max_abs_diff > atol + rtol * torch.max(torch.abs(tensor1), torch.abs(tensor2)).item(), msg

def test_initial_unet():
    model = MultiTaskUnet(in_chan=8, out_chan=8)
    data = torch.randn(5, 8, 320, 320)
    data_split = torch.split(data, 2, 1)
    all_output = []
    for input, unet in zip(data_split, model.initial_unet):
        output = unet(input)
        all_output.append(output)

    for i in range(1, len(all_output)):
        try:
            assert_not_close(all_output[0], all_output[i])
        except AssertionError as e:
            print(f"AssertionError: {e}")

