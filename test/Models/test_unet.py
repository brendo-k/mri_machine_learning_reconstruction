import pytest
import torch
from ml_recon.models.UNet import double_conv, down, up, concat, Unet, Unet_down, Unet_up

# Unit tests for double_conv module
def test_double_conv():
    in_chans = 3
    out_chans = 64
    drop_prob = 0.2
    input_shape = (1, in_chans, 256, 256)
    x = torch.randn(input_shape)

    # Create an instance of the double_conv module
    double_conv_module = double_conv(in_chans, out_chans, drop_prob, relu_slope=0.2)

    # Ensure the output shape matches the expected shape
    output = double_conv_module(x)
    assert output.shape == (1, out_chans, 256, 256)

# Unit tests for down module
def test_down():
    input_shape = (1, 64, 256, 256)
    x = torch.randn(input_shape)

    # Create an instance of the down module
    down_module = down()

    # Ensure the output shape matches the expected shape after downsampling
    output = down_module(x)
    assert output.shape == (1, 64, 128, 128)

# Unit tests for up module
def test_up():
    in_chan = 64
    out_chan = 32
    input_shape = (1, in_chan, 128, 128)
    x = torch.randn(input_shape)

    # Create an instance of the up module
    up_module = up(in_chan, out_chan)

    # Ensure the output shape matches the expected shape after upsampling
    output = up_module(x)
    assert output.shape == (1, out_chan, 256, 256)

# Unit tests for concat module
def test_concat():
    input_shape_encode = (1, 64, 256, 256)
    input_shape_decode = (1, 32, 128, 128)
    x_concat = torch.randn(input_shape_encode)
    x = torch.randn(input_shape_decode)

    # Create an instance of the concat module
    concat_module = concat()

    # Ensure the output shape matches the expected shape after concatenation
    output = concat_module(x, x_concat)
    assert output.shape == (1, 96, 128, 128)


# Test the forward pass of Unet model
def test_unet_forward():
    in_channels = 3
    out_channels = 1
    depth = 4
    chans = 18
    drop_prob = 0.2
    batch_size = 2
    height, width = 128, 128

    # Create an instance of the Unet model
    model = Unet(in_channels, out_channels, depth, chans, drop_prob, relu_slope=0.2)

    # Generate random input data
    input_data = torch.rand(batch_size, in_channels, height, width)

    # Perform forward pass
    output = model(input_data)

    # Check output shape
    assert output.shape == (batch_size, out_channels, height, width)

    # Check that all elements in the output are finite numbers
    assert torch.isfinite(output).all()

# Test the forward pass of Unet_down module
def test_unet_down_forward():
    in_channels = 18
    out_channels = 36
    drop_prob = 0.2
    batch_size = 2
    height, width = 64, 64

    # Create an instance of the Unet_down module
    down_module = Unet_down(in_channels, out_channels, drop_prob, relu_slope=0.2)

    # Generate random input data
    input_data = torch.rand(batch_size, in_channels, height, width)

    # Perform forward pass
    output = down_module(input_data)

    # Check output shape
    assert output.shape == (batch_size, out_channels, height // 2, width // 2)

    # Check that all elements in the output are finite numbers
    assert torch.isfinite(output).all()

# Test the forward pass of Unet_up module
def test_unet_up_forward():
    out_channels = 18
    in_channels = out_channels * 2
    drop_prob = 0.2
    batch_size = 2
    height, width = 64, 64

    # Create an instance of the Unet_down module
    up_module = Unet_up(in_channels, out_channels, drop_prob, relu_slope=0.2)

    # Generate random input data
    input_data = torch.rand(batch_size, in_channels, height, width)
    concat_data = torch.rand(batch_size, out_channels, height * 2 + 2, width * 2+ 5)

    # Perform forward pass
    output = up_module(input_data, concat_data)

    # Check output shape
    assert output.shape == (batch_size, out_channels, height * 2, width * 2)

    # Check that all elements in the output are finite numbers
    assert torch.isfinite(output).all()



if __name__ == '__main__':
    pytest.main()
