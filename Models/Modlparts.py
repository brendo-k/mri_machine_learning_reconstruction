from torch import nn 
import torch.functional as F

class parts(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        for _ in range(layers):
            self.conv_block.append(nn.Conv2d(64, 64, 3, padding=1))
            self.conv_block.append(nn.BatchNorm2d(64))
            self.conv_block.append(nn.ReLU())
            
        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 2, 3, padding=1), 
            nn.BatchNorm2d(2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.final_layer(x)
        return x