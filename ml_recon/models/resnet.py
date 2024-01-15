from torch import nn

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

class ResNet(nn.Module):
    def __init__(self, itterations, in_chan=2, out_chan=2, chans=32, scaling=0.1) -> None:
        super().__init__()

        self.cascade = nn.Sequential()
        for _ in range(itterations):
            self.cascade.append(residual_block(chans, scaling))
        self.cascade.append(nn.Conv2d(chans, chans, 3, padding=1, bias=False))

        self.encode = nn.Conv2d(in_chan, chans, 3, padding=1, bias=False)
        self.decode = nn.Conv2d(chans, out_chan, 3, padding=1, bias=False)
    

    def forward(self, x):
        x = self.encode(x)
        x = self.cascade(x) + x
        x = self.decode(x)
        return x

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--itterations', type=int, default=15, help='Number of residual blocks')
        return parser

