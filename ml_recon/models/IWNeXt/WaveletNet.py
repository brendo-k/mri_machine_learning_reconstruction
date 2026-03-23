import torch.nn as nn
from net.attention_ConvNext import Block


class WaveConvNeXt(nn.Module):
    def __init__(self, rank,n_channels, G0, kSize, D=4, C=4, G=32):
        super(WaveConvNeXt, self).__init__()
        self.rank=rank
        self.D = D   # number of RDB
        self.C = C   # number of Conv in RDB
        self.kSize = kSize  # kernel size

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(n_channels, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.SEDRDBs = nn.ModuleList()
        for i in range(self.D):
            self.SEDRDBs.append(
                Block(dim=32,rank=self.rank)
            )


        self.GFF = nn.Sequential(*[
            nn.Conv2d( G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])
        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G, n_channels//2, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

    def forward(self, x):
        f1 = self.SFENet1(x)
        x = self.SFENet2(f1)

        SEDRDBs_out = []
        for j in range(self.D):
            x = self.SEDRDBs[j](x)
            SEDRDBs_out.append(x)
        s_out=SEDRDBs_out[0]+SEDRDBs_out[1]+SEDRDBs_out[2]+SEDRDBs_out[3]
        x = self.GFF(s_out)
        x += f1

        output = self.UPNet(x)

        return output



