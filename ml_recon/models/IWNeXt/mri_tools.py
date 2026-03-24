import torch
import torch.fft

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)

def fftshift(x, axes=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    assert torch.is_tensor(x) is True#判断输入是否是张量
    if axes is None:
        axes = tuple(range(x.ndim()))#x.ndim()指的是x的维度,tuple指的是元组()。
        shift = [dim // 2 for dim in x.shape]#将x.shape对半输出,并且每个元素重复一个，输出[1，1，256，256]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)
    # return roll(x,shift,axes)

def ifftshift(x, axes=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    assert torch.is_tensor(x) is True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)
    # return roll(x,shift,axes)

def fft2(data):
    assert data.shape[-1] == 2
    data = torch.view_as_complex(data)
    data = torch.fft.fftn(data, dim=(-2, -1), norm='ortho')
    data = torch.view_as_real(data)
    data = fftshift(data, axes=(-3, -2))
    return data

def ifft2(data):
    assert data.shape[-1] == 2
    data = ifftshift(data, axes=(-3, -2))
    data = torch.view_as_complex(data)
    data = torch.fft.ifftn(data, dim=(-2, -1), norm='ortho')
    data = torch.view_as_real(data)
    return data

def rfft2(data):
    assert data.shape[-1] == 1
    data = torch.cat([data, torch.zeros_like(data)], dim=-1)
    data = fft2(data)
    return data


def rifft2(data):
    assert data.shape[-1] == 2
    data = ifft2(data)
    data = data[..., 0].unsqueeze(-1)
    return data


def rA(data, mask):#(1,256,256,1)
    assert data.shape[-1] == 1
    data = torch.cat([data, torch.zeros_like(data)], dim=-1)
    data = fft2(data) * mask
    return data


def rAt(data, mask):
    assert data.shape[-1] == 2
    data = ifft2(data * mask)
    data = data[..., 0].unsqueeze(-1)
    return data


def rAtA(data, mask):
    assert data.shape[-1] == 1
    data = torch.cat([data, torch.zeros_like(data)], dim=-1)
    # data= torch.cat([data, torch.ones_like(data)], dim=-1)
    data = fft2(data) * mask
    data = ifft2(data)
    data = data[..., 0].unsqueeze(-1)
    return data

def center_crop(data,shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to].contiguous()