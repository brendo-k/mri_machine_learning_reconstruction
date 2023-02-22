from Models.Unet import Unet
import torch

if __name__ == "__main__":
    net = Unet(1, 1)
    x = torch.rand((10, 1, 640, 320))
    net(x)
    parameters = sum(param.numel() for param in net.parameters())
    print(net)
    print(parameters)
