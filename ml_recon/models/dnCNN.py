import torch
import torch.nn as nn

class DnCNN(nn.Module):

    def __init__(self, 
                 in_chan, 
                 out_chan,
                 kernel_size = 3,
                 channels = 32,
                 num_of_layers = 15):
        super().__init__()

        self.dncnn = nn.Sequential()

        encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.dncnn.append(encoder)

        for _ in range(num_of_layers-2):
            self.dncnn.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=1, bias=False))
            self.dncnn.append(nn.InstanceNorm2d(channels))
            self.dncnn.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        decoder = nn.Conv2d(in_channels=channels, out_channels=out_chan, kernel_size=kernel_size, padding=1, bias=False)

        self.dncnn.append(decoder)

    def forward(self, x):
        return self.dncnn(x)
