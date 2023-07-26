import torch
from ml_recon.models.unet import double_conv, down, up, concat

# Unit tests for double_conv module
def test_double_conv():
    in_chans = 3
    out_chans = 64
    drop_prob = 0.2
    input_shape = (1, in_chans, 256, 256)
    x = torch.randn(input_shape)

    # Create an instance of the double_conv module
    double_conv_module = double_conv(in_chans, out_chans, drop_prob)

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
    x_encode = torch.randn(input_shape_encode)
    x_decode = torch.randn(input_shape_decode)

    # Create an instance of the concat module
    concat_module = concat()

    # Ensure the output shape matches the expected shape after concatenation
    output = concat_module(x_encode, x_decode)
    assert output.shape == (1, 96, 128, 128)