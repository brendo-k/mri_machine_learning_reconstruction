from Models import varnet
import torch

if __name__ == "__main__":
    x = torch.rand((10, 1, 640, 320), dtype=torch.complex64)
    mask = torch.ones((10, 1, 640, 320))
    net = varnet.VarNet(2, 2)
    net(x, mask)
