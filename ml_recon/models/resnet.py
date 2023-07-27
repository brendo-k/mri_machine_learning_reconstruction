from torch import nn
import torch 
from typing import Tuple

class residual_block(nn.Module):
    def __init__(self, chans, scaling) -> None:
        super().__init__()

        self.res_block = nn.Sequential(
            nn.Conv2d(chans, chans, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(chans),
            nn.ReLU(),
            nn.Conv2d(chans, chans, 3, stride=1, padding=1, bias=False),
        )
        self.scaling = scaling

    def forward(self, x):
        return self.res_block(x) * self.scaling + x

class resnet(nn.Module):
    def __init__(self, itterations, chans=32, scaling=0.1) -> None:
        super().__init__()

        self.cascade = nn.Sequential()
        for _ in range(itterations):
            self.cascade.append(residual_block(chans, scaling))
        self.cascade.append(nn.Conv2d(chans, chans, 3, padding=1, bias=False))

        self.encode = nn.Conv2d(2, chans, 3, padding=1, bias=False)
        self.decode = nn.Conv2d(chans, 2, 3, padding=1, bias=False)
    

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean


    def forward(self, x):
        x, mean, std = self.norm(x)
        x = self.encode(x)
        x = self.cascade(x) + x
        x = self.decode(x)
        x = self.unnorm(x, mean, std)
        return x