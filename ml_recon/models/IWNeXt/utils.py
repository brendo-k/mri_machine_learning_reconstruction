import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.nn.functional as F
import numpy as np

def psnr_slice(gt, pred, maxval=None):#输入是全采样图像和重建图像
    assert type(gt) == type(pred)#确保全采样图像和重建图像类型一致
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()#将Tensor格式转变成numpy格式
    batch_size = gt.shape[0]#赋值样本数
    PSNR = 0.0#初始化峰值信噪比
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval#返回一个浮点数
        PSNR += peak_signal_noise_ratio(gt[i].squeeze(), pred[i].squeeze(), data_range=max_val)#gt[i]指的是Groundtruth,pred[i]指的是输出图像
    return PSNR / batch_size

def ssim_slice(gt, pred, maxval=None):
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    batch_size = gt.shape[0]
    SSIM = 0.0
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval
        SSIM += structural_similarity(gt[i].squeeze(), pred[i].squeeze(), data_range=max_val)
    return SSIM / batch_size

def center_crop(data, shape):#中心裁剪用于数据增强
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]

def normalize_zero_to_one(data, eps=0.):#数据归一化层
    data_min = float(data.min())
    data_max = float(data.max())
    return (data - data_min) / (data_max - data_min + eps)

def D(p1,p2,z1,z2): # negative cosine similarity
        return - (F.cosine_similarity(p1, z1, dim=1).mean()+F.cosine_similarity(p2, z2, dim=1).mean())*0.5

def D1(p, z): # negative cosine similarity
    z = z.detach() # stop gradient
    p = F.normalize(p, dim=1) # l2-normalize
    z = F.normalize(z, dim=1) # l2-normalize
    return 1-(p*z).sum(dim=1).mean()
def D2(p,z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    print(F.cosine_similarity(p, z.detach(), dim=1))
    return 1- F.cosine_similarity(p, z.detach(), dim=1).mean()
def calc_ssl_loss(u, v):
    scalar=0.5
    abs_u_minus_v = torch.abs(u - v)#取绝对值
    abs_u = torch.abs(u)
    term_1 = torch.sqrt(torch.sum(torch.pow(abs_u_minus_v, 2))) / torch.sum(torch.sqrt(torch.pow(abs_u, 2)))#2范数
    term_2 = torch.sum(abs_u_minus_v) / torch.sum(abs_u)#1范数
    return scalar*(term_1 + term_2)

def complex2real(data, axis=-1):
    assert type(data) is np.ndarray
    data = np.stack((data.real, data.imag), axis=axis)
    return data

def real2complex(data, axis=-1):
    assert type(data) is np.ndarray
    assert data.shape[axis] == 2
    mid = data.shape[axis] // 2
    data = data[..., 0:mid] + data[..., mid:] * 1j
    return data.squeeze(axis=axis)